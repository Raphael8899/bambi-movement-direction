"""Shared paths, dataset coordinates and constants. Import with `from config import ...`
(run notebooks/scripts from the project root)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# This machine has an Application Control policy that blocks compiled DLLs loaded
# from under Desktop\, so a venv created inside this project can't import numpy.
# We run with the interpreter below instead; requirements.txt reproduces the env
# on a normal machine.
INTERPRETER = r"C:/Users/rapha/AutoCode/bambi-analysis/.venv/Scripts/python.exe"

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
ANNOTATIONS_DIR = PROJECT_ROOT / "annotations"   # gold labels, kept in git
CROPS_DIR = OUTPUT_DIR / "crops"

# Reuse the dataset that's already on disk (12,655 images) instead of re-downloading.
# Override with the BAMBI_DATA_DIR environment variable if it moves.
_EXISTING = Path(r"C:/Users/rapha/AutoCode/bambi-analysis/data/bambi-dataset")
_LOCAL = PROJECT_ROOT / "data" / "bambi-dataset"
DATASET_DIR = Path(os.getenv("BAMBI_DATA_DIR",
                             str(_EXISTING if _EXISTING.exists() else _LOCAL)))

for _d in (OUTPUT_DIR, ANNOTATIONS_DIR, CROPS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Roboflow project (only needed to re-download the dataset). v1 has different class ids.
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_WORKSPACE = "bambi-overview"
ROBOFLOW_PROJECT = "bambi-alfs-20250520-upload04-sdakr"
ROBOFLOW_VERSION = 2

SPLITS = ["train", "valid", "test"]
IMG_SIZE = 2048

# Roboflow remapped the original class ids 2/3/4 to 0/1/2. The data.yaml names
# ('2','3','4') are misleading, so use this mapping. NOTE: the id->species mapping
# is an UNCONFIRMED assumption carried over from an old project -- the dataset stores
# only ids 0/1/2; confirm Rotwild/Rehwild/Schwarzwild with the BAMBI team.
CLASS_NAMES = {0: "Rotwild", 1: "Rehwild", 2: "Schwarzwild"}
CLASS_EN = {0: "red deer", 1: "roe deer", 2: "wild boar"}
CLASS_COLORS = {0: (255, 80, 80), 1: (80, 255, 80), 2: (80, 80, 255)}

# filenames look like 12_8083_jpg.rf.<hash>.jpg -> flight 12, frame 8083
FILENAME_RE = r"^(\d+)_(\d+)_jpg"

SEED = 42
FIGURE_DPI = 150

# Heading is in degrees, 0 = east (+x), clockwise on screen (image y points down, so
# 90 = south), pointing at the head. Axial quantities (body/blur axis) live in [0,180).
# The annotation direction codes are defined in src/annotation/label_store.py.

CROP_PAD = 0.20
NORM_CROP_SIZE = (96, 96)
