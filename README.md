# BAMBI — Movement Direction Estimation of Wildlife from Thermal Light-Field Drone Imagery

Computer Vision course project (FH Hagenberg) · Raphael & Andreas.

Estimating the **movement direction** of wildlife (red deer, roe deer, wild boar) in thermal
**Airborne Optical Sectioning (AOS)** light-field drone imagery:
- **Branch S** — body *orientation* of stationary animals.
- **Branch M** — motion-*blur direction* of moving animals.

A comparative study of classical image processing, classical ML, CNNs (from scratch),
transfer learning (DINOv2), and a foundation-model negative control (BioCLIP) — validated with
manual gold labels, tracking-derived pseudo-ground-truth, and proper circular-statistics evaluation.

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

## Planned pipeline
EDA → BB refinement → crops/quality → movement classification → classical orientation/blur →
DL bake-off → tracking → evaluation. Each step is a notebook that reads `src/` and writes `output/`.

## Tests
```bash
.venv/Scripts/python -m pytest tests/ -q
```

## Layout
- `src/` — library code (data loading, BB refinement, features, orientation/, blur/, tracking, circular metrics, annotation tool)
- `notebooks/` — phase notebooks (read `src/`, write `output/`)
- `annotations/` — **gold labels** (kept in git)
- `output/` — derived artifacts (gitignored)
- `docs/` — plans, references, report drafts
