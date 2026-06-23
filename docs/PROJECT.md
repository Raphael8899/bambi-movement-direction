# BAMBI Movement Direction Estimation - project documentation

Technical overview of the project: the task, the dataset, the approach, the results, and how to
reproduce them. The headline numbers here are all recomputed from the committed result tables in
`output/` (see `results.md`, `validation.md`, and the critical self-review in `audit.md`).

## 1. The task
A Computer Vision course project at FH Hagenberg, based on the BAMBI thermal light-field drone wildlife
data. The work covers preprocessing, annotation, and applying, evaluating and comparing computer-vision
methods. Our chosen topic: estimate the **movement direction (heading)** of wildlife in thermal
Airborne Optical Sectioning (AOS) drone imagery. Detection and a moving/stationary call already exist in
BAMBI; the heading is the open question.

## 2. The dataset (verified empirically)
- Roboflow `bambi-overview/bambi-alfs-20250520-upload04-sdakr`, version 2, in YOLO format. The original
  images are not redistributed here (see "Data" in the README); we add only our own labels.
- 2048x2048 thermal AOS integral images; 12,655 image files (12,514 with at least one box); 46,046
  boxes; 223 distinct flight ids (221 with boxes), about 65 frames each.
- Animal crops are median 65 px on the long side, and 67 % of images contain more than one animal -
  small, low-contrast, crowded targets.
- Classes are stored only as ids **0 / 1 / 2** (the data.yaml names them '2' / '3' / '4'). A mapping to
  red deer / roe deer / wild boar is an unverified assumption and is not relied on; everything uses
  the class id.
- The export has no drone GPS or pose; some frames show AOS border ghosting (about 0.9 % flagged).

## 3. The approach
There is no human-verifiable ground truth for direction: the head is rarely visible on the warm blobs,
most crops are multi-animal, and hand-labelling direction would just be a second guess. So the **ground
truth for direction comes from tracking**: register consecutive frames to remove the drone's ego-motion,
then the animal's residual displacement over the track is its heading. On top of that we apply, evaluate
and **compare** method families under a leakage-free, flight-disjoint, circular-statistics protocol.
A set of 1,500 manually labelled crops serves as an independent validation set, not as the ground truth.

Pipeline: AOS data -> classical preprocessing (tighten boxes onto the warm blob) -> tracking (the
direction ground truth) -> comparison of single-image and learned methods -> evaluation.

## 4. Results (label-free core; full detail in results.md, caveats in audit.md)
- **Tracking recovers a real heading.** Consecutive registration removes the large drone ego-motion and
  leaves a coherent animal residual; the static background cancels in the per-step difference images.
  Of 2,697 tracklets, 190 pass the confidence gate, and the defensible high-confidence core is **138**
  (at least 8 steps and 50 px net displacement). Median registration inlier ratio 0.86.
- **Single-image direction is weak.** The gradient structure tensor (GST) is the best single-image
  estimator at 29.1 deg median axial error, but it only barely beats - and on the most common class
  loses to - a constant per-class-heading prior. It is not a substitute for tracking.
- **Moving vs stationary from one crop is modest.** Frozen DINOv2 / CLIP / BioCLIP features reach
  0.62-0.64 balanced accuracy, classical hand features 0.58, a from-scratch CNN 0.50; all sit below the
  honest scene-structure ceiling (about 0.84), so part of the signal is scene correlation.
- **EDA:** per-class size and intensity statistics; the low refined-area ratio on class 2 is a
  segmentation effect on the coolest, lowest-contrast animals, not loose boxes.

## 5. Validation against the manual labels (see validation.md)
On the 1,500 manually labelled crops:
- **Tracking direction is confirmed:** human axis vs tracking axis about 22.7 deg median (19.1 on
  clearly-moving crops), Acc@45 0.79-0.86, versus about 50 deg for chance.
- **The head is recognisable in only 14 % of crops** (27 % among movers) - the quantitative reason the
  direction ground truth comes from tracking, not hand labels.
- **GST matches the human body axis well** (about 10.7 deg), but signed movement direction stays hard.
- **Moving vs stationary on the real labels** (flight-disjoint): about 0.62 balanced accuracy, below the
  0.78 scene ceiling - the same conclusion as the proxy labels. No part of the pipeline had to change.

## 6. Honest limitations
- The tracker is a simple greedy nearest-centroid associator with no appearance model, so in crowded
  scenes two same-class animals can be swapped mid-track (ID switch). The confidence gate limits this
  but cannot remove it; this is the main weakness and is stated openly in `audit.md`.
- Single-image methods recover the body axis but not the head/tail sign (180-degree ambiguity).
- The class-id-to-species mapping is unverified.
- The overlap between the trusted tracks and the manual labels is small (n = 56): consistent and clearly
  above chance, but not a large sample.

## 7. Repository guide
- `config.py` - paths, dataset coordinates, class map, constants.
- `src/` - data loading, classical box refinement, crops, GST, blur estimators, circular statistics,
  image registration, tracking, direction, movement features, and frozen deep features; plus the
  annotation tool under `annotation/`.
- `scripts/` - the runnable pipeline steps (EDA, tracking, single-image evaluation, the movement
  bake-off, label validation, the annotation-package builder) and `verify_claims.py`.
- `tests/` - unit tests (136 passing): `python -m pytest tests/ -q`.
- `output/` - the committed result tables (CSV); figures are produced separately, not stored here.
- `annotations/labels.csv` - the 1,500 manually labelled crops (the added data).

## 8. Reproduce
Set up the environment (`requirements.txt`), point `BAMBI_DATA_DIR` at the dataset, then run the scripts
in `scripts/` (details and the run order are in the README). `python scripts/verify_claims.py`
re-derives every headline number from the committed CSVs without needing the raw images.
