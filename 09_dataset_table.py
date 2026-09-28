"""Write DATASET.md: every xeno-canto recording used, with its difficult-condition tags."""
import pandas as pd

import config as C


def main():
    rec = pd.read_csv(C.ROOT / "recordings.csv")
    clips = pd.read_csv(C.DATA / "clips.csv")
    per_rec = clips.groupby("xc_id").agg(clips=("species", "size"), faint=("faint", "sum"), noisy=("noisy", "sum"))
    rec = rec.join(per_rec, on="xc_id").fillna({"clips": 0, "faint": 0, "noisy": 0})

    lines = ["# Dataset", "",
             "All audio from [xeno-canto.org](https://xeno-canto.org) via API v3. Licences are per recording "
             "(Creative Commons, see each link); recordists are credited in `recordings.csv`.", "",
             "Condition tags per recording: **faint** / **noisy** = number of its 5 s clips in the quietest / "
             "lowest-SNR quartile of the whole dataset; **background** = other species the recordist listed "
             "(overlapping calls).", ""]
    summary = rec.groupby("species").agg(recordings=("xc_id", "size"), clips=("clips", "sum"),
                                         faint_clips=("faint", "sum"), noisy_clips=("noisy", "sum"),
                                         with_background_species=("n_background", lambda s: int((s > 0).sum())))
    lines += ["| species | recordings | clips | faint clips | noisy clips | recordings with background species |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {sp} | {r.recordings} | {int(r.clips)} | {int(r.faint_clips)} | {int(r.noisy_clips)} | "
              f"{r.with_background_species} |" for sp, r in summary.iterrows()]
    missing = [sp for sp in C.CLASSES if sp not in summary.index]
    lines += [f"| {sp} | 0 | 0 | - | - | - |" for sp in missing] + [""]
    if missing:
        lines += [f"{', '.join(missing)} is a *restricted species* on xeno-canto (a cage-bird-trade target): the API "
                  "redacts the audio file and location for all of its recordings, so no audio could be obtained. "
                  "See the write-up.", ""]
    for sp, g in rec.groupby("species", sort=False):
        lines += [f"## {sp} ({g.scientific.iloc[0]})", "",
                  "| XC id | quality | type | clips | faint | noisy | background species |", "|---|---|---|---|---|---|---|"]
        for r in g.sort_values("xc_id").itertuples():
            bg = r.background_species if isinstance(r.background_species, str) else ""
            lines.append(f"| [XC{r.xc_id}]({r.url}) | {r.quality} | {r.type if isinstance(r.type, str) else ''} | "
                         f"{int(r.clips)} | {int(r.faint)} | {int(r.noisy)} | {bg} |")
        lines.append("")
    (C.ROOT / "DATASET.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
