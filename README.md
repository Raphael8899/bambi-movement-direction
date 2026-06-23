# BAMBI: Movement Direction Estimation of Wildlife from Thermal Light-Field Drone Imagery

Computer Vision course project (CVI4IL, FH Hagenberg).

## Overview

We estimate the **movement direction** of wild animals (class ids 0/1/2; an assumed red deer / roe
deer / wild boar mapping) in thermal **Airborne Optical Sectioning (AOS)** light-field drone imagery.

The hard part: there is no ground truth for direction, because the head is almost never visible on the
small, blurry warm blobs. Our central idea is to take the truth from motion over time rather than from a
single image. We track each animal across the frames of a flight, register consecutive frames to remove
the drone's ego-motion, and the animal's residual displacement is its heading. This tracking-derived
heading is our ground truth. On top of that we tried several computer-vision approaches and recorded how
well they do, under a leakage-free, flight-disjoint, circular-statistics protocol. We hand-labelled 1,500 crops as an independent **validation** set, not as the truth.

## Key results (all reproducible from the committed result tables)

- Tracking ground truth: 2,697 tracklets, 190 pass the confidence gate, 138 form the defensible
  high-confidence core. Median registration inlier ratio 0.86.
- Single-image direction is a weak, class-dependent cue: GST is the best estimator at 29.1 deg median
  axial error, but a per-class mean-heading prior (25.0 deg) is competitive overall.
- Moving vs stationary from a single crop is modest: frozen foundation features reach about 0.62 to 0.64
  balanced accuracy, classical hand features 0.58, all below the honest scene-structure ceiling (~0.84).
- Human-label validation confirms the approach: human axis vs tracking axis ~22.7 deg median (19.1 deg
  on clear movers) versus ~50 deg for chance; GST matches the human body axis well (~10.7 deg); and the
  head is recognisable in only 14 % of crops, which is exactly why we take direction from tracking.

Full numbers are in `docs/results.md` and `docs/validation.md`. Run `python scripts/verify_claims.py` to
re-derive every headline number from the committed CSVs.

## Repository layout

- `src/` library code: data loading, bounding-box refinement (classical segmentation), crops, GST,
  blur estimators, circular statistics, image registration, tracking, direction, movement features,
  frozen deep features, and the annotation tool.
- `scripts/` runnable pipeline steps and `verify_claims.py` (the self-checker).
- `tests/` unit tests (136 passing).
- `annotations/labels.csv` the additional labelled data produced in this project.
- `output/` the result tables (CSV). Figures are intentionally not in the repo (see "Figures and
  presentation").
- `docs/` documentation: `PROJECT.md` (project overview), `results.md`, `validation.md`,
  `audit.md` (critical self-review), and `references.md`.
- `config.py` paths, dataset coordinates and constants. `requirements.txt` the Python environment.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # add torch (cu121) for the deep-learning part
```

The code expects the dataset on disk (see "Data"). Point to it with the `BAMBI_DATA_DIR` environment
variable, or let `config.py` resolve the default local path. Re-downloading from Roboflow needs a
`ROBOFLOW_API_KEY` in a local `.env` file (never committed).

## Data

The images are the public Roboflow dataset `bambi-overview / bambi-alfs-20250520-upload04-sdakr`,
version 2 (12,655 images, 46,046 bounding boxes, 223 flights):
https://universe.roboflow.com/bambi-overview/bambi-alfs-20250520-upload04-sdakr/dataset/2

We did **not** modify or re-upload the original images or their bounding boxes. The boxes came with the
dataset in YOLO format; we did not train a detector. The only data we added is the manual labelling in
`annotations/labels.csv` (1,500 curated single-animal crops), with columns:

- `crop_id` of the form `{split}_{flight}_{frame}_{line}` (e.g. `train_98_988_0`). It maps a label to
  the original image and to the `line`-th bounding box in that image's YOLO label file.
- `motion_state` one of `stationary`, `moving`, `unsure` (legacy `slight`/`moderate`/`strong` still
  parse and collapse to `moving`).
- `direction_class` the annotated axis/head code; `direction_deg` the resulting angle (0 = up,
  clockwise; see `docs/validation.md` for the conversion to the tracking frame).
- `annotator`, `timestamp_iso`.

The class ids 0/1/2 are the dataset's own labels; the species names (red deer / roe deer / wild boar)
are an unverified assumption and are not relied on anywhere in the analysis - everything uses the class id.

## How to reproduce

Run from the project root with the project interpreter. The main steps:

```bash
python scripts/eda.py                 # dataset statistics
python scripts/run_tracking.py        # tracking ground truth -> output/tracking_directions.csv
python scripts/eval_blur.py           # single-image axis estimators vs tracking
python scripts/movement_experiment.py # moving/stationary bake-off (classical, CNN, frozen backbones)
python scripts/validate_labels.py     # validation against the human labels (needs the raw images)
python scripts/verify_claims.py       # re-derive and check every headline number from the CSVs
python -m pytest tests/ -q            # unit tests
```

`verify_claims.py` runs without the raw images: it only needs the committed CSVs plus pandas and numpy.

## Model weights

This project is mostly classical computer vision plus frozen, pretrained backbones, so there are no
large trained weights to distribute:

- The detector was not trained (the boxes came with the dataset).
- Segmentation, registration, tracking, GST and the blur estimators are classical (no learning).
- The foundation models (DINOv2, CLIP, BioCLIP via `open_clip` / `torch`) are used **frozen**: we only
  read their feature vectors, we do not fine-tune them.
- The only trained components are a small logistic-regression / random-forest head and a from-scratch
  CNN (which did not converge on this small, weak data). They are tiny and are produced on the fly when
  you run `scripts/movement_experiment.py`; no checkpoint files are needed.

## Figures and presentation

The slide deck is included as `docs/presentation.pptx` (also uploaded to the course e-learning).
Standalone figures are not stored separately; the numbers they are built from are the committed CSVs in
`output/`, so every figure can be regenerated and checked against the data.

## Honesty notes and limitations

- Direction ground truth comes from tracking; human labels only validate it.
- The tracker is a simple greedy nearest-centroid associator (no Kalman/SORT). In crowded scenes it can
  swap two nearby animals (ID switch); the confidence gate mitigates but does not remove this.
- Single-image methods recover the body/blur axis but not the head/tail sign (180-degree ambiguity).
- The species-name mapping is unverified.
- The trusted-vs-human validation overlap is small (n = 56): consistent and clearly above chance, but
  not a large sample.

## Documentation

Start with `docs/PROJECT.md` (project overview: task, dataset, approach, results, repo guide). Then
`docs/results.md` (numbers), `docs/validation.md` (the human-label check), `docs/audit.md` (a critical
self-review of what could be overstated and why), and `docs/references.md` (literature).
