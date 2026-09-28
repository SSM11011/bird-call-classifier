"""Query xeno-canto (API v3), pick a varied set of recordings per species, download them.

Selection is deliberately *not* "best quality only": recordings are drawn
round-robin from buckets of (quality rating A-E) x (has background species or not),
so faint/noisy low-rated recordings and recordings with other birds calling
are represented alongside clean ones.

Restricted species (Oriental Magpie-Robin, a cage-bird-trade target) have their
audio withheld by the API. If you obtain such recordings legitimately, drop them
in data/manual/<English name>/XC<id>.mp3; they are picked up here, their metadata
is fetched by catalogue number, and the class is trained like any other.

Writes data/raw/XC<id>.mp3 and recordings.csv (the dataset manifest).
"""
import json
import random
import re
import shutil
import time
from collections import defaultdict

import pandas as pd
import requests
from tqdm import tqdm

import config as C

API = "https://xeno-canto.org/api/3/recordings"


def fetch_all(query: str) -> list[dict]:
    recs, page, pages = [], 1, 1
    while page <= pages:
        r = requests.get(API, params={"query": query, "key": C.api_key(),
                                      "per_page": 500, "page": page}, timeout=60)
        r.raise_for_status()
        d = r.json()
        pages = int(d["numPages"])
        recs += d["recordings"]
        page += 1
        time.sleep(1)
    return recs


def seconds(length: str) -> int:
    s = 0
    for part in length.split(":"):
        s = s * 60 + int(part)
    return s


def select(recs: list[dict], cap: int, rng: random.Random, rare: bool) -> list[dict]:
    usable = [r for r in recs
              if r.get("file") and "_meta" not in r  # restricted species: audio is withheld
              and r.get("status", "identified") == "identified"
              and (rare or seconds(r["length"]) <= C.MAX_RECORDING_SECONDS)]
    buckets = defaultdict(list)
    for r in usable:
        buckets[(r["q"] or "no score", bool(r["also"]))].append(r)
    for b in buckets.values():
        rng.shuffle(b)
    chosen = []
    keys = sorted(buckets)
    while len(chosen) < cap and any(buckets.values()):
        for k in keys:
            if buckets[k] and len(chosen) < cap:
                chosen.append(buckets[k].pop())
    return chosen


def main():
    rng = random.Random(C.SEED)
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for en, sci in C.SPECIES.items():
        recs = fetch_all(f'sp:"{sci}"')
        chosen = select(recs, C.MAX_RECORDINGS_PER_SPECIES, rng, rare=en in C.RARE_CLASSES)
        restricted = sum("_meta" in r for r in recs)
        manual = sorted((C.DATA / "manual" / en).glob("*.mp3"))
        for path in manual:
            xc_id = re.search(r"XC(\d+)", path.name).group(1)
            meta = fetch_all(f"nr:{xc_id}")[0]
            meta["file"] = str(path)
            shutil.copy(path, C.RAW_DIR / f"XC{xc_id}.mp3")
            chosen.append(meta)
        print(f"{en:24s} available={len(recs):5d} restricted={restricted:4d} "
              f"manual={len(manual):3d} selected={len(chosen)}")
        for r in chosen:
            rows.append({
                "xc_id": int(r["id"]), "species": en, "scientific": sci,
                "url": r["url"], "quality": r["q"], "length": r["length"],
                "type": r["type"], "country": r["cnt"], "background_species": "; ".join(r["also"]),
                "n_background": len(r["also"]), "remarks": " ".join((r["rmk"] or "").split())[:300],
                "recordist": r["rec"], "license": r["lic"], "file": r["file"],
            })

    df = pd.DataFrame(rows)
    for row in tqdm(df.itertuples(), total=len(df), desc="download"):
        out = C.RAW_DIR / f"XC{row.xc_id}.mp3"
        if out.exists() and out.stat().st_size > 0:
            continue
        for attempt in range(3):
            try:
                with requests.get(row.file, timeout=120, stream=True) as resp:
                    resp.raise_for_status()
                    out.write_bytes(resp.content)
                break
            except requests.RequestException as e:
                print(f"XC{row.xc_id}: {e} (attempt {attempt + 1})")
                time.sleep(5)
    df["downloaded"] = [(C.RAW_DIR / f"XC{i}.mp3").exists() for i in df.xc_id]
    df.drop(columns="file").to_csv(C.ROOT / "recordings.csv", index=False)
    print(df.groupby("species")[["downloaded"]].sum())
    print(json.dumps(df.groupby("species").quality.value_counts().unstack(fill_value=0).to_dict(), indent=1))


if __name__ == "__main__":
    main()
