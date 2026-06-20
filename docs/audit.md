# Critical review (self-audit)

A skeptical pass over the label-free results, checking for confirmation bias, confounds and
overstatement. Everything here is recomputed from the output CSVs with the repo scripts; no
number is taken on trust.

## 1. Are the 190 tracking "directions" real, or registration artefacts?
Mostly real. Trusted tracklets have a median net displacement of 292 px against a ~1 px
registration residual (signal/noise ~290x), so the heading reflects genuine animal motion, not
drift. Caveat: 11 of 190 sit near the displacement floor (10-25 px) and are weaker; the other
179 are solid. The deeper check - does the tracking heading match what a human sees - is still
pending (Andreas's labels).

## 2. "Movers' blur agrees better than stationary" was a confound, not evidence
We had read the ~11 deg gap as proof the blur carries direction. That reading is wrong.
Stationary animals have no real direction; their "ground truth" is the tracker's noisy heading,
which is near-uniform (axial R = 0.10). Any estimator necessarily disagrees with that noise, so
the gap is largely tautological. Crop sizes are matched (movers 69 px vs stationary 65 px), so
it is at least not a size artefact - but it is not evidence, and we drop it.

## 3. Does single-image direction estimation beat a trivial prior?
This is the honest bar, and the answer is mostly no. A constant-heading baseline (predict the
per-class mean heading, ignore the image) is strong because headings are not uniform:

| class | GST median err | constant-baseline median err | GST helps? |
|---|---|---|---|
| 0 (most common, 380 crops) | 39.4 | 16.1 | no - worse than the prior |
| 1 | 25.4 | 32.6 | yes |
| 2 | 19.3 | 36.2 | yes |
| overall | 29.1 | 30.6 | barely (+1.5 deg) |

GST only adds value for classes 1 and 2, whose headings are dispersed; for class 0 the direction
is essentially predictable from the class alone and the image-based estimate is worse than
guessing the mean. (The constant baseline is computed in-sample, which if anything flatters it;
the picture holds either way.) Revised claim: single-frame blur/shape is a weak, class-dependent
cue that helps only where headings are not already predictable - not "the blur signal is real".

## 4. Foundation models "win" moving/stationary - real but overstated
The three frozen foundation models are statistically indistinguishable: balanced accuracy
DINOv2 0.633 +/- 0.079, CLIP 0.619 +/- 0.068, BioCLIP 0.643 +/- 0.036 - differences well inside
one std, so "BioCLIP best" is noise. They beat the majority baseline (0.50) and modestly edge
the hand-feature random forest (0.584 +/- 0.045, about one std). But the ceiling is low
(AUC ~0.70) on weak tracker-derived proxy labels, the proxy is partly circular with the tracker,
and the from-scratch CNN collapsed (data-limited, not a fair "thermal beats CNNs"). Net:
foundation features are competitive here, but the result is modest, proxy-limited, and not a
clean test of the domain gap. Re-run on human labels before drawing conclusions.

## 5. Smaller caveats
- The EDA "refined area = 0.37 of the box" for class 2 may partly reflect the segmentation
  grabbing only the warm core of a low-contrast boar, not purely loose boxes - needs an eyeball check.
- The +16% tracking yield (164 -> 190) is real (same gate, better registration); the claim that
  the dropped tracklets were "false positives" is a plausible interpretation, not independently proven.

## What still stands
- Tracking gives genuine movement directions for clearly-moving animals (S/N ~290x).
- Most animals are stationary and correctly rejected; direction is undefined for them.
- Single-image direction estimation is weak; tracking is the right ground truth.
- Among single-image methods, GST > FFT/cepstrum (a stable ranking).
These are the defensible claims; the rest is preliminary and waits on human labels.
