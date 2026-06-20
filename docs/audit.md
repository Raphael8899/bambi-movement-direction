# Critical review (self-audit)

A skeptical, end-to-end pass over the label-free work: confirmation bias, confounds,
overstatement, leakage, and number errors. Every figure here was recomputed from the output
CSVs, cross-checked by an independent adversarial review, and the tracking claim was checked
visually on real frames. Nothing is taken on trust.

## What survives scrutiny (the defensible claims)
- **Tracking gives genuine movement directions for clearly-moving animals.** Checked several ways:
  (a) ego-motion removal is real - on a single-animal mover (no ID-switch possible) consecutive
  ORB+RANSAC registration (median inlier 0.95) cancels ~800 px of cumulative drone translation and
  leaves a coherent 205 px animal residual that matches the pipeline output exactly; in the per-step
  difference images the static forest background cancels to near-black and only the moving animal
  leaves a residual. (b) the drone-artefact hypothesis is refuted - stationary tracklets within a
  flight do NOT share a drift direction (in the flights re-tested that have >=3 stationary animals,
  none cluster beyond a uniform null), and in a mixed flight the stationary animals read 7-23 px net
  while movers read 159-803 px. The honest signal/noise is ~30-40x at the NET level (mover median net
  292 px vs stationary median net 7 px; equivalently coherent 292 px vs the ~2.5 px/step registration
  noise random-walking to ~9 px over a track). Per step a mover exceeds the noise floor by only ~3x;
  what makes a heading trustworthy is its COHERENCE (R, Rayleigh) across many steps, not raw per-step
  magnitude. Note that 30-40x is movers vs *clearly-stationary* animals - a curated comparison; trusted
  vs ALL untrusted tracklets (which include incoherent / failed tracks at ~146 px median net) separate
  by only ~2x, so this is per-track signal quality, not population-level separability.
- **Most animals are stationary and correctly rejected**; their "headings" are near-uniform noise.
- **GST > FFT/cepstrum/gradient/moment** for single-image axis (a stable ranking); the blur
  convention is clean (adding +45/+90 deg only increases error - no hidden offset).
- **Single-image direction is weak** - it barely beats (overall) or loses to (class 0) a constant
  per-class-heading prior. Correctly and honestly stated.
- **Foundation models are statistically indistinguishable** from each other; "BioCLIP best" is noise.
- The 164 -> 190 / inlier 0.72 -> 0.86 registration improvement reproduces exactly.

## Problems found (corrections)

### 1. "190 trusted directions" overstates - the gate is too lenient
The trusted gate (n_steps>=5, R>=0.5, disp>=5 px) lets through a weak tail:
- `corr(R, n_steps) = -0.29`: high coherence is partly trivial when there are few steps. 23 of
  190 have <=6 steps (median R there 0.89).
- 21 of 190 have net displacement < 50 px. High R with tiny displacement (e.g. R=0.99 at 34 px)
  is the dangerous case - a small constant registration bias mimics motion.
- Together **42 of 190 are suspect**. The **defensible high-confidence core is 138** (n_steps>=8
  AND disp>=50 px; classes 58/49/31; median disp 327 px). Report 138, not 190.

### 2. ID-switch confound spans the whole crowded core (not just 18 tracklets)
~67% of (frame, class) groups contain more than one same-class animal (median 3, p95 13), and 91%
of detections share their frame with a same-class sibling. The tracker uses class + nearest-centroid
only, no appearance model, with a generous gap-scaled gate (120 px+). The R/Rayleigh gate is the ONLY
defense against switches and cannot tell a coherent switch chain from a single animal. This is broader
than first stated: the trusted set lives almost entirely in crowded scenes - the median trusted (and
core) tracklet sits in a frame with **~7-8 same-class animals**, and only **6 of 190 trusted (3 of 138
core)** are genuine single-animal tracks. All **18 tracklets with displacement > 1000 px fall inside
the 138 core**. So the heading statistics inherit ID-switch uncertainty across the whole core, not only
the 18 extreme tracklets. Example: flight 223 has 9 trusted boars all heading 28-72 deg (max disp
1650 px, no stationary animals to anchor registration) - plausibly a real moving herd, but
indistinguishable from ID-switching across a row of boars. The visual ego-motion check above is itself
caveated: TID 444 is a single-animal track (one of the ~3% switch-immune cases), so it proves
ego-motion removal but is not representative of the crowded population the headings are computed on.

### 3. Moving/stationary experiment: scene-ID leakage inflates the bar
The proxy labels are heavily flight-clustered: **mean per-flight label purity 0.89, and 112 of
164 flights are 100% one label**. With whole flights held out, "predict this flight's majority"
already reaches ~0.84 balanced accuracy using no per-animal motion. So the honest reference is
the scene structure, not the 0.50 global baseline. The models score 0.62-0.64 - clearly below
that scene ceiling (so not just memorising flights), but the result cannot cleanly claim "a
single crop detects motion"; it is partly scene correlation. Weak-but-real, more confounded than
the docs first said. (Class is not the confound: P(moving|class) = 0.44/0.38/0.47.)

### 4. The boar "refined area = 0.37 of the box" is segmentation failure, not loose boxes
The warm-blob segmenter under-segments low-contrast boars (the coolest class, intensity 79):
the segmented blob is a median ~7% of the crop (vs ~13-20% for the other classes), and the metric
is unstable (0.26-0.37 across samples). So 0.37 means the segmenter grabs only the warm core of a
boar, NOT that the manual
boxes are especially loose. The earlier "manual boxes most oversized for class 2" reading is
wrong and is corrected in results.md.

### 5. "Movers agree better than stationary" was a confound (kept from the first pass)
Stationary animals have no real direction; their "gt" is near-uniform noise (axial R=0.10), so
any estimator disagrees with it by construction. The ~11 deg gap is tautological, not evidence.
Dropped. (Crop sizes are matched - movers 69 px vs stationary 65 px - so at least not a size effect.)

### 6. Number errors in the docs (now fixed)
- class-0 mover crops: was "380", actual **415**. Medians 39.4/16.1 are correct; only n was wrong.
- "945 crops from the 190 movers": 945 crops but only **189 tracklets** produced crops, and the
  945 are pseudo-replicated (5 crops share one per-tracklet gt), so there are **189 independent
  directions**, not 945 - point estimates are fine (tracklet-level GST median 28.7 ~ 29.1), but
  any confidence interval on 945 would be ~5x too tight.
- images: "12,655" total files vs **12,514** with >=1 box. (Flights: **223 distinct flight ids total,
  221 with >=1 box** - both correct; "223" is not an error but the total, e.g. the real flight 223 in #2.)
- random blur baseline median 46.1 / Acc@45 0.48 -> 44.2 / 0.50 on recompute (RNG-draw, not load-bearing).

### 7. The first pass's "~290x signal/noise" was itself wrong (corrected above)
It divided a cumulative net displacement (292 px) by a single-step residual ("0.75 px") - mixed
units, and the "0.75 px" floor does not reproduce (the per-step registration noise is ~2.5 px median,
heavy-tailed). Honest separation: ~3x per step, ~30-40x at the net level (coherent 292 px vs noise
random-walking to ~9 px over a track). The ego-motion removal itself was re-verified independently
(consecutive registration cancels ~800 px of drone motion to leave a coherent 205 px residual on a
single-animal mover; background cancels in the difference images), so the conclusion holds - only the
ratio was inflated.

### 8. Per-class direction concentration must be measured per tracklet, not per crop
The blur eval pools 945 crops (5 per tracklet). Computing R / Rayleigh on those pseudo-replicates
inflates significance ~5x. On the 190 independent trusted tracklet headings the honest figures are:
axial non-uniformity overall p~1.3e-4 (R 0.22); per class axial R = 0.55 / 0.23 / 0.12 (class 0
clustered, class 2 dispersed - which is why the constant-axis baseline beats GST on class 0 and loses
on class 2). Note the non-uniformity is AXIAL: directionally (head/tail) only class 1 has a net
heading (p~3e-4); the overall directional distribution is ~uniform (p~0.18).

## Bottom line
The core is real and was checked, not asserted: tracking recovers genuine movement for clearly
moving animals (re-verified here from scratch - consecutive registration removes ~800 px of drone
motion and leaves a coherent residual, background cancels in the difference images), and single-image
direction estimation is honestly weak. The overstatements were (a) "190" should be ~138
high-confidence, (b) the ID-switch confound spans the whole crowded core (median ~7-8 same-class
animals/frame), not just 18 tracklets, (c) the movement experiment's bar
is scene-structure, not 0.50, and (d) the eye-catching "~290x signal/noise" was a unit error - the
real separation is ~30-40x net / ~3x per step. None of these flip a headline, but they make the
honest story narrower and the numbers exact. Final validation still waits on Andreas's human labels.
