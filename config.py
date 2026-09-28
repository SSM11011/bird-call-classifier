"""Shared configuration for the bird call classifier pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"            # original xeno-canto mp3s
RESULTS = ROOT / "results"

# English name -> scientific name as used by xeno-canto and by Perch's label list.
SPECIES = {
    "Indian Cuckoo": "Cuculus micropterus",
    "Indian Eagle-Owl": "Bubo bengalensis",
    "Oriental Magpie-Robin": "Copsychus saularis",
    "Purple-rumped Sunbird": "Leptocoma zeylonica",
    "Zitting Cisticola": "Cisticola juncidis",
    "Red-vented Bulbul": "Pycnonotus cafer",
}
CLASSES = list(SPECIES)

# Download: cap the common species so the run stays small, take every usable
# recording of the rare one. The remaining imbalance is intentional.
MAX_RECORDINGS_PER_SPECIES = 60
MAX_RECORDING_SECONDS = 600       # skip hour-long soundscapes for common species

# Audio / windowing (Perch v2 expects 5 s of 32 kHz mono audio).
SR = 32_000
CLIP_SECONDS = 5.0
CLIP_SAMPLES = int(SR * CLIP_SECONDS)
HOP_SECONDS = 2.5
MAX_AUDIO_SECONDS = 300           # only scan the first 5 min of long files
CLIPS_PER_RECORDING = 8
CLIPS_PER_RECORDING_RARE = 16     # more windows from each Eagle-Owl recording
RARE_CLASSES = {"Indian Eagle-Owl"}
NOISE_CLIPS_PER_RECORDING = 2     # quietest windows, reused as real background noise

N_FOLDS = 5
SEED = 0


def api_key() -> str:
    """Read the xeno-canto key from the environment or a local, gitignored .env."""
    key = os.environ.get("XC_API_KEY")
    env_file = ROOT / ".env"
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("XC_API_KEY="):
                key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("Set XC_API_KEY (see README).")
    return key
