# BAMBI — Movement Direction Estimation of Wildlife from Thermal Light-Field Drone Imagery

Computer Vision course project (FH Hagenberg) · Raphael & Andreas.

Estimating the **movement direction** of wildlife (class ids 0/1/2 — an *assumed* red deer / roe
deer / wild boar mapping) in thermal **Airborne Optical Sectioning (AOS)** light-field drone imagery.

There is no ground truth for direction (the head is rarely visible on the warm blobs), so the
**direction ground truth comes from tracking**: register consecutive frames to cancel the drone's
ego-motion, and the animal's residual displacement is its heading. On top of that we **compare**
methods — classical single-image axis estimators (GST, FFT, cepstrum, …), classical ML, a
from-scratch CNN, and frozen foundation models (DINOv2, CLIP, BioCLIP) — under leakage-free,
flight-disjoint, circular-statistics evaluation. Andreas's 1,500 human labels are the independent
**validation** set, not the ground truth.

**Start with [docs/PROJECT.md](docs/PROJECT.md)** — the single ground-truth doc (task, dataset,
decisions, current verified state, repo guide). The full plan is in
[docs/plans/2026-06-19-movement-direction-design.md](docs/plans/2026-06-19-movement-direction-design.md);
results in [docs/results.md](docs/results.md); the critical self-review in [docs/audit.md](docs/audit.md).

## Setup
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # add torch==2.x+cu121 for the DL phase
```
The dataset (Roboflow `bambi-alfs-20250520-upload04-sdakr` v2, 12,655 images) is reused from a local
path by default; see `config.py` (`BAMBI_DATA_DIR` to point elsewhere). Re-downloading needs
`ROBOFLOW_API_KEY` in `.env`.

## Annotation
`scripts/build_annotation_package.py` produces `dist/bambi_annotation.zip` — a self-contained
labelling tool plus the crops to label. It runs anywhere with Python 3 and Pillow; see the README
inside the package. Labels come back as `labels.csv` and go under `annotations/`.

## Pipeline
EDA → BB refinement → crops/quality → tracking (direction ground truth) → classical orientation/blur
→ movement classification → DL bake-off → label validation → evaluation. Each step is a script under
`scripts/` that reads `src/` and writes `output/`.

## Tests
```bash
.venv/Scripts/python -m pytest tests/ -q
```

## Layout
- `src/` — library code (data loading, BB refinement, features, blur, GST, tracking, circular metrics, annotation tool)
- `scripts/` — phase scripts (read `src/`, write `output/`) + `verify_claims.py`
- `annotations/` — Andreas's validation `labels.csv` (kept in git)
- `output/` — derived artifacts (result CSVs + figures committed; crops/caches gitignored)
- `docs/` — plan, references, results, audit, validation, presentation
