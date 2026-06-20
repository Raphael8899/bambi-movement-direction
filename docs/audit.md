# Critical review (self-audit)

A skeptical, end-to-end pass over the label-free work: confirmation bias, confounds,
overstatement, leakage, and number errors. Every figure here was recomputed from the output
CSVs, cross-checked by an independent adversarial review, and the tracking claim was checked
visually on real frames. Nothing is taken on trust.

## What survives scrutiny (the defensible claims)
- **Tracking gives genuine movement directions for clearly-moving animals.** Checked two ways:
  (a) visually - in registered frames the animal walks away from its start marker while the
  background stays fixed (e.g. trusted tracklets at 266 px and 506 px); (b) the drone-artefact
  hypothesis is refuted - stationary tracklets within a flight do NOT share a drift direction
  (only 2 of 24 flights cluster beyond a uniform null's 95th percentile), and in a mixed flight
  the stationary animals read 7-23 px while movers read 159-803 px. Signal/noise ~290x (trusted
  median net displacement 292 px vs stationary per-step residual ~0.75 px).
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

### 2. ID-switch confound in crowded flights (not previously mentioned)
75% of (frame, class) groups contain more than one same-class animal (median 3, p95 14). The
tracker uses class + nearest-centroid only, no appearance model, with a generous 120 px+ gate.
The R gate rejects incoherent switches, but **18 trusted tracklets have displacement > 1000 px**,
and in a crowd a switch chain that happens to walk one way yields a coherent but fake "heading".
Example: flight 223 has 9 trusted boars all heading 28-72 deg (max disp 1650 px, no stationary
animals to anchor registration) - plausibly a real moving herd, but indistinguishable from
ID-switching across a row of boars. The high-displacement trusted tracklets are an uncontrolled
confound; treat them with caution.

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
the segmented blob is a median ~5% of the crop, and the metric is unstable (0.26-0.37 across
samples). So 0.37 means the segmenter grabs only the warm core of a boar, NOT that the manual
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
- flights: "223" -> **221** valid flight ids; images "12,655" total files vs **12,514** with >=1 box.
- random blur baseline median 46.1 / Acc@45 0.48 -> 44.2 / 0.50 on recompute (RNG-draw, not load-bearing).

## Bottom line
The core is real and was checked, not asserted: tracking recovers genuine movement for clearly
moving animals, and single-image direction estimation is honestly weak. The overstatements were
(a) "190" should be ~138 high-confidence, (b) an undisclosed ID-switch confound in crowds, and
(c) the movement experiment's bar is scene-structure, not 0.50. None of these flip a headline,
but they make the honest story narrower and the numbers exact. Final validation still waits on
Andreas's human labels.
