# BAMBI — Movement Direction Estimation · Master Design

**Course:** Computer Vision (FH Hagenberg). **Team:** Raphael & Andreas (2 people).
**Grading weight:** 50 % project. **Effort:** ~40–50 h/student. **Deliverables:** presentation + report + reproducible code repo.
**Date:** 2026-06-19. **Status:** approved (scope, methods, decisions).

> Research question (fixed by proposal): *Can we determine the movement direction of wildlife
> from thermal light-field (AOS) drone imagery?* Methodology is ours to design.

## Status updates (2026-06-19)
After Andreas trialled the annotation tool, two things changed from the original plan:
- **Ground truth for direction now comes from tracking, not hand-labelling.** Hand-labelling
  direction proved unreliable (67% of crops have more than one animal; most animals are faint or
  compact and the head is not visible; there is no truth a human can verify). Tracking the animal
  across a flight's frames, after removing drone ego-motion, gives an objective heading (validated on
  flight 98: per-step resultant R=0.86, Rayleigh-significant). Human labels become a small validation
  set on clearly visible animals (does tracking match what a person sees), plus the moving/stationary
  call and the "how often is the head even recognisable" rate. The annotation package therefore
  focuses on moving/standing first, direction only when the head is obvious.
- **Species is shown as the dataset class id (0/1/2), not a name.** The names Rotwild/Rehwild/
  Schwarzwild are not in the dataset (data.yaml names the classes '2','3','4'); the id->species
  mapping came from the old project and is unverified - to be confirmed with the BAMBI team. It does
  not affect the direction task.

## Approved decisions
- **Label schema:** 8 directional classes (45° sectors) **+ "axis-only / unsure" escape** + moving/stationary flag.
- **Species focus:** start with **Rotwild (red deer)** for deep-learning depth; classical methods on all 3.
- **Annotation budget:** ≤ **5 h** of Raphael's time → lean on tracking pseudo-GT + fast custom tool.
- **Start:** build P0–P2 autonomously, notify when feasibility batch is ready.

## Empirical ground truth about the data (verified, not assumed)
- Roboflow `bambi-overview/bambi-alfs-20250520-upload04-sdakr` v2; **2048×2048**; **12,655 imgs / 46,046 boxes**
  (Rotwild 21,787 / Rehwild 17,403 / Schwarzwild 6,856); **223 flight ids (221 with boxes) × ~65 frames**.
- Crop longest-side **median 65 px** (only 2.6 % < 32 px) → orientation feasible (proposal's "20–40 px" was wrong).
- **94 %** of warm blobs have a clear elongation axis → body/blur **axis (mod 180°)** recoverable even classically.
- **No drone GPS/pose metadata** in the export → ego-motion must be image-based; proposal's "combine with GPS" dropped.
- Cross-frame displacement is dominated by **drone survey ego-motion** (rectangular trajectories) → tracking needs
  frame-to-frame registration; after the pivot above this registered tracking became the **primary direction GT**.
- AOS integration registers source frames to the ground plane → within one image the **blur is ground-relative**
  (the cleanest moving-animal signal). [Bimber/Schedl/Kurmi AOS lit.]
- Existing `bambi-analysis` repo: movement labels are **unvalidated** sharpness clustering (confounded by background
  texture); "oriented bbox" angle stored-but-unused. Sanity-check only; re-derive & validate independently.

## Scientific framing → why it earns the grade
A **comparative study** of CV methods (covers all four lecture families) on a real, hard task, with **rigorous,
honest evaluation**. Anchored in the supervisors' own group: **IAOS** (Nathan, Kurmi, Bimber, *Drones* 2022,
arXiv:2207.13344) explicitly frames moving targets as directional motion blur in AOS integrals.

Two sub-tasks:
- **Branch S** — stationary animals → body orientation (axis primary; head/tail stretch).
- **Branch M** — moving animals → motion-blur direction (axis primary; sign via tracking/thermal asymmetry).

## Method bake-off (identical flight-disjoint splits, matched budgets, multi-seed)
| Family | Branch S | Branch M | Role |
|---|---|---|---|
| Classical IP | PCA/moments/`fitEllipse`, **GST**+coherence | **GST**, gradient-orientation histogram, **Radon-of-spectrum**, cepstrum (baseline), IAOS-GLV framing | interpretable lower bound, no domain gap, label-free |
| Classical ML | hand features (Hu, HOG, elongation, intensity profile) → SVM/RF (8 bins) | blur-length/coherence → moving/stationary RF | M/S classifier + bin classification |
| CNN from scratch | small CNN, 1-ch stem, **biternion** (cos2θ,sin2θ) + **von Mises loss** | axis + optional length | "no-pretraining" arm (often strong on thermal) |
| Transfer | ImageNet CNN partial fine-tune; **frozen DINOv2 + head** | dito | tests if RGB priors help |
| Foundation model | **BioCLIP/CLIP zero-shot = negative control** | — | quantifies domain gap |

**Moving/Stationary classifier (rebuilt):** GST elongation/coherence + inner-vs-surround sharpness (animal-specific) +
blur length; validated against human labels (unsupervised vs supervised comparison).

## Ground-truth strategy (no GPS collars) — revised after the feasibility pivot
1. **Tracking displacement = primary direction GT** for moving animals (Rayleigh-filtered, after
   ego-motion registration). Hand-labelling direction proved unreliable (see Status updates above), so
   the original "gold human labels = primary" plan was dropped.
2. **Human labels = validation, not primary:** a small set on clearly-visible animals (does tracking
   match perception?), the moving/stationary call, and the head-discernibility rate.
3. **Internal validators (GT-free):** within-track temporal consistency; body-axis ↔ blur-axis agreement; cross-method agreement.

## Evaluation protocol
- Wrapped circular distance; P=360 directional, P=180 axial (angle-doubling).
- Metrics: MAE/median/RMSE/P90; **Acc@{10,22.5,45}°**; **flip-corrected** (axis vs sign). Stratify by motion×species×size.
- Mandatory baselines: uniform-random, constant-mean (after Rayleigh), **permutation test** p-values.
- **Flight-disjoint splits** (no leakage); **block-bootstrap CIs at flight level**.
- Tracking agreement: Jammalamadaka–Sarma circular correlation + permutation p + Bland-Altman (moving subset only).

## Tracking branch (sign + validation)
Per flight: image-based registration (ORB/LK + RANSAC **partial-affine / similarity**) → SORT with **centroid/Kalman-predicted** gating
(IoU fails on small fast blobs) → heading via circular stats + Rayleigh filter. Honest caveats (no pose, ghosted borders).

> Update (post-implementation): the SORT/Kalman gating was **dropped** for a simple greedy
> nearest-centroid associator - deliberate (honesty pillar 3). It is enough on these short
> tracklets and avoids a tuned motion model; see `src/tracking.py`.

## Risks & fallbacks
| Risk | Fallback |
|---|---|
| head/tail not recoverable | axis (mod 180°) is the complete headline result; head/tail = negative result + tracking sign |
| tracking too noisy | restrict to the high-confidence core (>=8 steps, >=50 px) + internal consistency; human validation set anchors it |
| DL overfits | classical methods strong standalone; the comparison itself is the contribution |
| low annotation yield | classical arms need 0 labels; ~300 labels suffice for a test set |
| ghosting/quality | quality filter; report retained fraction |

## Phase plan (semester)
P0 setup · P1 BB-refinement+quality+crops · P2 annotation tool + **feasibility batch (~100–150, Raphael ~0.5 h)** ·
P3 M/S + classical branches · P4 **validation set (~1,500, Andreas)** · P5 DL bake-off · P6 tracking+sign ·
P7 evaluation/stats · P8 report + presentation.

## Annotation spec (~5 h, done by Andreas)
One stratified batch of ~1,500 crops (red deer 600 / roe deer 500 / boar 400), visibility-filtered and
shuffled so any prefix is representative. Per crop: motion state (stationary / moving / unsure) and facing
direction (8 compass classes, or axis-only when head/tail is ambiguous, or none). Honest "unsure" over
guessing. Shipped as a standalone package (`dist/bambi_annotation.zip`) that runs anywhere with Python +
Pillow; output is `labels.csv`. A small re-labelled subset can give an inter-annotator agreement figure.

## Implementation notes
Clean Python repo; reuse the local dataset; re-derive all results rather than trusting the old pipeline;
TDD for the core math (circular stats, GST, registration, metrics); fixed seeds and a documented run order;
the independent model variants in the bake-off run as separate jobs.

## Key references
- IAOS — Nathan, Kurmi, Bimber, *Drones* 2022 (arXiv:2207.13344)
- Through-Foliage Tracking with AOS — Nathan et al., *J. Remote Sensing* 2022 (arXiv:2111.06959)
- Thermal AOS — Kurmi, Schedl, Bimber, *Remote Sensing* 2019 (MDPI 11/14/1668)
- Biternion Nets — Beyer et al., GCPR 2015 · Deep Directional Statistics — Prokudin et al., ECCV 2018
- Circular Smooth Label — Yang & Yan, ECCV 2020 · GWD loss — Yang et al., ICML 2021
- Drone-thermal wildlife detection benchmark — arXiv:2310.11257 · BioCLIP camera-trap failure — Springer 2025
- Full annotated bibliography: `docs/references.md` (to be written in P7).
