# Results so far

Status of the label-free part of the project (before Andreas's manual labels arrive). All
numbers were recomputed independently from the output CSVs. The objective ground truth is the
improved tracking output (190 directions); the earlier 164-direction run is kept at
`output/tracking_directions_v1.csv`.

## Dataset (measured)
- 2048x2048 thermal AOS integrals; 12,655 image files (12,514 with >=1 box), 46,046 boxes;
  classes 0/1/2 with 21,787 / 17,403 / 6,856 boxes; 221 flights with boxes (223 distinct flight ids total) x ~65 frames.
- Animal crops: median longest side 65 px. 67% of images contain more than one animal (mean 3.7).
- The dataset only stores a class id (0/1/2); the species names are not in it (confirm the
  id->species mapping with the BAMBI team).

## Per-class statistics (`scripts/eda.py`)
How oversized the manual boxes are (refined warm-blob area / manual box area) and intensity:

| class | n boxes | crop long px (median) | refined/manual area | median intensity |
|---|---|---|---|---|
| 0 | 21,787 | 69 | 0.81 | 102 |
| 1 | 17,403 | 61 | 1.15 | 116 |
| 2 |  6,856 | 61 | 0.37 |  79 |

- Class 0 boxes are oversized (refined/box 0.81); class 1 boxes are tight (1.15 - the smallest
  animals fill their box). The class-2 (boar) value 0.37 is NOT loose boxes but a SEGMENTATION
  FAILURE: boars are the coolest, lowest-contrast class, so the warm-blob segmenter grabs only
  their core (blob ~7% of the crop, vs ~13-20% for classes 0/1). See [audit.md](audit.md).
- Class 2 is the coolest / darkest class (median intensity 79 vs 102 / 116).
- A border-ghosting heuristic (`src/quality.py`) flags ~0.9% of frames at the scan's default
  threshold (1.5); the top-scoring ones are real AOS sheared-border artefacts on inspection. It is a
  tunable triage heuristic, not a validated detector.

## Tracking-derived movement direction (the objective ground truth)
Pipeline (`src/registration.py` with CLAHE + Lowe ratio test, `src/tracking.py`,
`src/direction.py`, `scripts/run_tracking.py`): register consecutive frames on the background
(ORB + RANSAC partial-affine / similarity) to cancel drone ego-motion, take the animal's residual displacement, and
aggregate per tracklet into a heading with a confidence (resultant length R + Rayleigh test).

- Over all flights: 2,697 tracklets; **190 pass the trusted gate** (83 / 68 / 39 by class), but
  the gate is lenient - ~42 are weak (<=6 steps with trivially-high R, or <50 px displacement
  that could be drift). The **defensible high-confidence core is 138** (>=8 steps and >=50 px;
  58/49/31). Median registration inlier ratio 0.86.
- Caution (ID-switch): ~67% of frames have >1 same-class animal (91% of detections share a
  same-class sibling) and the tracker has no appearance model. The trusted set is almost entirely
  in crowded scenes - the median trusted tracklet sits among ~7-8 same-class animals/frame and only
  ~3% are single-animal tracks - so a coherent switch chain could mimic a heading across the whole
  core, not only the 18 tracklets with >1000 px displacement. The R/Rayleigh gate is the only
  defense; the visual ego-motion check used an atypical single-animal flight. See [audit.md](audit.md).
- Most animals are stationary -> no coherent heading -> correctly rejected. The movers are
  clear (median net displacement ~300 px, R ~0.66, ~11 steps).
- Better registration (CLAHE + ratio test) raised the yield from 164 to 190 (+16%) and the
  median inlier from 0.72 to 0.86 without loosening the gate. The tracklets that dropped out
  were mostly previous false positives - apparent coherence that was actually registration drift.

## Do single-image cues recover that direction?
Axis comparison (mod 180), 945 crops from 189 mover tracklets (so 189 independent directions,
not 945 - the crops are pseudo-replicated), error in degrees:

| method (single image) | median err | Acc@45 |
|---|---|---|
| GST (structure tensor) | 29.1 | 0.68 |
| spectrum (Radon)       | 32.8 | 0.61 |
| cepstrum               | 33.1 | 0.60 |
| gradient histogram     | 33.5 | 0.59 |
| moments / PCA          | 35.2 | 0.59 |
| random baseline        | 44.2 | 0.50 |

- Among single-image methods GST (spatial-domain) ranks above the FFT / cepstrum ones on these
  small low-contrast crops (a stable ranking).
- The honest bar is a constant-heading baseline (predict the per-class mean heading, ignore the
  image), strong because the headings are not uniform. GST vs that baseline (median axial error):

  | class | GST | constant baseline |
  |---|---|---|
  | 0 (415 crops) | 39.4 | 16.1 (GST worse than the prior) |
  | 1 | 25.4 | 32.6 |
  | 2 | 19.3 | 36.2 |
  | overall | 29.1 | 30.6 |

  So single-image estimation only helps for classes 1 and 2 (dispersed headings); for the most
  common class 0 the direction is predictable from the class alone and the image estimate is
  worse than guessing the mean.
- The earlier "movers agree better than stationary" figure is NOT used as evidence: stationary
  animals have no real direction (their gt is near-uniform noise, R=0.10), so that gap is
  tautological. See [audit.md](audit.md).

Verdict: single-frame blur/shape is a weak, class-dependent cue that helps only where headings
are not already predictable from the class. Tracking is the right ground truth; single-image
methods do not replace it.

## Moving vs stationary from a single crop (`scripts/movement_experiment.py`)
Weak proxy labels from tracking (moving = trusted directions, stationary = low-displacement
tracklets; tracker-derived, so partly circular - to be re-checked against human labels).
Flight-disjoint GroupKFold, 5 folds x 3 seeds:

| feature family | balanced acc | ROC-AUC |
|---|---|---|
| frozen BioCLIP | 0.64 | 0.70 |
| frozen DINOv2  | 0.63 | 0.69 |
| frozen CLIP    | 0.62 | 0.67 |
| hand features (random forest) | 0.58 | 0.62 |
| CNN from scratch | 0.50 | 0.56 |
| majority baseline | 0.50 | - |

- The three frozen foundation models beat the majority baseline and modestly edge the hand-feature
  random forest (~1 std), but they are statistically indistinguishable from each other (balanced
  acc 0.62-0.64, within one std - so "BioCLIP best" is noise). The ceiling is low (AUC ~0.70) on
  noisy, partly circular tracker-derived proxy labels, and the from-scratch CNN collapsed
  (data-limited, not a clean "thermal beats CNNs"). The proxy labels are also flight-clustered
  (per-flight purity 0.89), so the honest bar is the scene structure (~0.84 balanced acc), not the
  0.50 baseline - the models sit below it, so this is partly scene-correlation, not proven
  single-crop motion detection. Modest and proxy-limited - re-check with human labels. See
  [audit.md](audit.md).

## Human-label validation (DONE — full detail in [validation.md](validation.md))
Andreas labelled all 1,500 crops (`annotations/labels.csv`):
- Human axis vs tracking axis: **~22.7 deg median** (19.1 on human-confirmed movers), Acc@45
  0.79-0.86, vs ~50 deg random → the tracking direction GT is **confirmed**.
- GST vs the human axis: **~10.7 deg median, Acc@45 0.90** — GST is a good *orientation* estimator;
  signed movement direction stays the hard part (GST vs tracking-movement ~29 deg).
- Head visible only **14 %** of the time (27 % among movers) → confirms the tracking pivot.
- Moving/stationary on real labels (flight-disjoint): LogReg 0.62, below the 0.78 scene ceiling.

Still open: an optional direction-regression DL arm (data-limited), then the written report.
