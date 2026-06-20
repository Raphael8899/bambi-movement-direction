# Results so far

Status of the label-free part of the project (before Andreas's manual labels arrive). All
numbers were recomputed independently from the output CSVs. The objective ground truth is the
improved tracking output (190 directions); the earlier 164-direction run is kept at
`output/tracking_directions_v1.csv`.

## Dataset (measured)
- 2048x2048 thermal AOS integrals; 12,655 images, 46,046 boxes; classes 0/1/2 with
  21,787 / 17,403 / 6,856 boxes; 223 flights x ~65 frames.
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

- Manual boxes are most oversized for class 2 (refined area ~37% of the box) and class 0
  (0.81); class 1 boxes are already tight (1.15 - the smallest animals fill their box).
- Class 2 is the coolest / darkest class (median intensity 79 vs 102 / 116).
- A border-ghosting heuristic (`src/quality.py`) flags ~0.9% of frames at its default
  threshold; the top-scoring ones are real AOS sheared-border artefacts on inspection. It is a
  tunable triage heuristic, not a validated detector.

## Tracking-derived movement direction (the objective ground truth)
Pipeline (`src/registration.py` with CLAHE + Lowe ratio test, `src/tracking.py`,
`src/direction.py`, `scripts/run_tracking.py`): register consecutive frames on the background
(ORB + RANSAC affine) to cancel drone ego-motion, take the animal's residual displacement, and
aggregate per tracklet into a heading with a confidence (resultant length R + Rayleigh test).

- Over all flights: 2,697 tracklets; **190 with a trustworthy direction** (83 / 68 / 39 by
  class); median registration inlier ratio 0.86.
- Most animals are stationary -> no coherent heading -> correctly rejected. The movers are
  clear (median net displacement ~300 px, R ~0.66, ~11 steps).
- Better registration (CLAHE + ratio test) raised the yield from 164 to 190 (+16%) and the
  median inlier from 0.72 to 0.86 without loosening the gate. The tracklets that dropped out
  were mostly previous false positives - apparent coherence that was actually registration drift.

## Do single-image cues recover that direction?
Axis comparison (mod 180), 945 crops from the 190 movers, error in degrees:

| method (single image) | median err | Acc@45 |
|---|---|---|
| GST (structure tensor) | 29.1 | 0.68 |
| spectrum (Radon)       | 32.8 | 0.61 |
| cepstrum               | 33.1 | 0.60 |
| gradient histogram     | 33.5 | 0.59 |
| moments / PCA          | 35.2 | 0.59 |
| random baseline        | 46.1 | 0.48 |

- Among single-image methods GST (spatial-domain) ranks above the FFT / cepstrum ones on these
  small low-contrast crops (a stable ranking).
- The honest bar is a constant-heading baseline (predict the per-class mean heading, ignore the
  image), strong because the headings are not uniform. GST vs that baseline (median axial error):

  | class | GST | constant baseline |
  |---|---|---|
  | 0 (380 crops) | 39.4 | 16.1 (GST worse than the prior) |
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
  (data-limited, not a clean "thermal beats CNNs"). Modest and proxy-limited - re-check with human
  labels. See [audit.md](audit.md).

## Pending (needs Andreas's manual labels)
- Validate the tracking direction against human perception; report the head-discernibility rate.
- Re-run the moving/stationary and direction evaluations against human labels (not the proxy).
- A direction-regression deep-learning arm trained on labels + tracking, on identical
  flight-disjoint splits.
