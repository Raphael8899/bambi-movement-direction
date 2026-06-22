# Validation against human labels

Andreas labelled all 1,500 curated crops (`annotations/labels.csv`). This is the independent
check the whole project was waiting on. `scripts/validate_labels.py` produces every number here
from the labels + the tracking output + the raw images, writing the per-crop
`output/label_validation.csv` and the aggregate `output/label_validation_summary.csv`;
`scripts/verify_claims.py` then re-checks those aggregates without needing the raw dataset.

**Convention caveat (important):** the annotation tool stores angles as 0 = up, clockwise; the
tracking heading is `atan2(dy, dx)` with y down (0 = east). They differ by 90 deg, so the human
angle is mapped to the tracking frame with **-90 deg** before any comparison. Sanity check: the
human-vs-tracking axis error is **67.3 deg without** the conversion and **22.7 deg with** it - so
the conversion is right and the agreement below is real, not an artefact.

## What we got
- 1,500 labels. Motion: unsure 652 · stationary 550 · slight 256 · moderate 40 · strong 2.
- Direction: nothing (-2) 537 · axis-only 749 · head (full arrow) 214.

## 1. The linchpin: does the human agree with the tracking direction?
On crops where tracking *trusted* a direction **and** the human drew a line, comparing the axes
(mod 180, human mapped to the tracking frame):

| subset | n | median err | Acc@45 |
|---|---|---|---|
| all trusted-mover crops with a line | 56 | **22.7 deg** | 0.79 |
| + human also says "moving" | 22 | **19.1 deg** | 0.86 |
| + single-animal frame (cleanest) | 5 | 19.1 deg | 0.80 |
| random baseline | - | 49.8 deg | 0.45 |

**The tracking ground truth is confirmed by independent human perception** (~20 deg median,
far above chance). Where the human marked a head (n=7 on trusted movers), the tracking sign was
correct 100% of the time. Caveat: the overlap is modest (the curated crops were not specifically
the trusted movers), so this validates the direction on a small but consistent, clearly-above-chance
sample.

## 2. Single-image orientation (GST) vs the human axis
On **all** 963 crops where the human drew a line (mostly stationary body orientation):

- GST vs human axis: **median 10.7 deg, Acc@45 0.90** (n=963); on the 214 head crops, median 8.2 deg.

So the GST single-image estimator is a **good orientation estimator** - it reads the same body/blur
axis a human sees. This is a *different and easier task* than recovering the **movement direction**:
against the tracking movement axis on movers, GST is only ~29 deg and barely beats a constant prior
(see `results.md`). Orientation is easy; signed movement direction is the hard part.

## 3. Head-discernibility rate (this is itself a result)
- Overall: a full head was given for only **14 %** of crops; axis-only 50 %; nothing 36 %.
- Among human-"moving" crops: head 27 %, axis-only 65 %, nothing 8 %.

A human can almost never tell head from tail on these thermal blobs - **exactly the reason we took
the direction ground truth from tracking instead of from hand labels.** The labels confirm that
premise quantitatively.

## 4. Moving vs stationary on the REAL labels (flight-disjoint)
Replacing the weak tracking proxy with Andreas's labels (moving = slight/moderate/strong, n=298;
stationary n=550; 169 flights):

- LogReg **0.62**, Random Forest 0.58 balanced accuracy; majority 0.50; per-flight-majority
  scene-ceiling 0.78 (per-flight label purity 0.87 averaged over flights; 0.82 crop-weighted).

Single-crop moving/stationary stays **modest and below the scene ceiling** - the same conclusion as
with the proxy labels, now confirmed on real labels. Note "slight" (256) is borderline and noisy.

One honest mismatch worth stating: the human calls many animals "moving" (mostly "slight") that the
strict tracking gate does **not** trust (sensitivity 0.08). The trusted set is **high-precision,
low-recall** by design - it does not catch every mover, but it is right about direction on the ones
it does keep (section 1). That is the correct reading, not a contradiction.

## 5. How many motion levels? (decided from the data)
The tool offered four intensity levels; Andreas used **stationary 550 · slight 256 · moderate 40 ·
strong 2** (and 652 "unsure" = 43 %). We tested whether the fine levels carry signal:
- They do **not** track objective tracking displacement (the per-tracklet net displacement is
  ID-switch-confounded and even runs slightly *backwards* across the levels).
- In image features only the stationary/moving split shows up (inner sharpness ~6.8 stationary vs
  ~4.5 for all movers; blur length flat).
- Learnability (flight-disjoint, hand features): **stationary vs moving** = bal-acc 0.62 / AUC 0.66;
  but **slight vs moderate+strong (within movers) = 0.52, i.e. chance**; a 3-level scale = 0.43
  (chance 0.33); the "clearly moving" class in any merge is only ~42 crops.

**Decision: collapse to a binary scale - `stationary` vs `moving` (= slight+moderate+strong) - plus
`unsure`.** The finer intensity is noise, not signal. The existing labels are remapped with
`merge_motion()` (no re-labelling needed; legacy levels still parse), and the tool now offers only
`s` stationary / `d` moving / `u` unsure. Note: 43 % "unsure" is itself a result - humans often cannot
even decide moving vs standing on these crops.

## Verdict
The human labels **confirm the project**: tracking recovers a real movement direction (humans agree
~20 deg), single-image GST captures orientation well (~11 deg) but not signed direction, and the head
is rarely visible (14 %) - which is precisely why tracking, not hand-labelling, is the ground truth.
No part of the pipeline had to be redesigned. Remaining limitation: the trusted/human direction
overlap is small, so the direction validation is consistent but not high-n.
