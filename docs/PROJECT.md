# BAMBI Movement Direction Estimation - project ground truth

Single source of truth. Read this first; it tells a fresh reader (or a fresh session) what the
project is, why, what was decided, what is actually true so far, and where everything lives.

Read order: this file -> `source/assignment.md` (the brief) + `source/proposal_decoded.txt` (our
proposal slides) -> `results.md` (numbers) -> `validation.md` (human-label check) -> `audit.md` (what is overstated and why) ->
`plans/2026-06-19-movement-direction-design.md` (the plan) -> `references.md` (literature).

## 1. The task
University Computer Vision project (FH Hagenberg), 50% of the grade, team of 2 (Raphael + Andreas),
~40-50 h/student. Based on the BAMBI thermal light-field drone wildlife data. We must preprocess,
annotate, and **apply + evaluate + compare** CV methods, then present.
**Our fixed topic:** estimate the **movement direction** of wildlife in thermal AOS drone imagery.
Methodology is ours. Full brief + supervisor email facts: `source/assignment.md`.

## 2. Our proposal, in brief (decoded slides in source/proposal_decoded.txt)
Light-field (AOS) integral images: a stationary animal stays sharp, a moving animal smears in its
direction of travel. The existing BAMBI pipeline detects animals and classifies moving vs stationary
but not *which* direction. Proposal: estimate direction in two cases - (a) body orientation of
stationary animals (deep learning), (b) motion-blur direction of moving animals (signal processing) -
and cross-check with tracking + drone data. Proposal feared 20-40 px crops and no ground truth.

## 3. The dataset (verified empirically, not assumed)
- Roboflow `bambi-overview/bambi-alfs-20250520-upload04-sdakr` v2, already downloaded at
  `C:\Users\rapha\AutoCode\bambi-analysis\data\bambi-dataset` (YOLO format) - reuse, don't re-download.
- 2048x2048 thermal AOS integrals; 12,655 image files (12,514 with >=1 box); 46,046 boxes; 221 flights
  with boxes (223 distinct flight ids total)
  x ~65 frames. Crops are median 65 px (NOT 20-40 as feared). 67% of images have >1 animal.
- Classes are stored only as ids **0/1/2** (data.yaml names them '2','3','4'). The names
  Rotwild/Rehwild/Schwarzwild are NOT in the dataset - the id->species map is an unverified
  assumption from an old project; **confirm it with the BAMBI team.** The tool/manifests show "class N".
- No drone GPS/pose/telemetry in the export. Some frames have AOS border ghosting (~0.9% flagged).
- Don't trust the old `C:\Users\rapha\AutoCode\bambi-analysis` repo's results (its own notes admit
  unvalidated movement labels etc.) - re-derive and validate.

## 4. The pivotal methodology decision
Hand-labelling direction proved unreliable (most animals are faint/compact, head vs tail not visible,
no human-verifiable ground truth, 67% multi-animal crops). So the **ground truth for direction comes
from TRACKING**, not human labels: register consecutive frames to remove drone ego-motion, then the
animal's residual displacement is its real heading. Human annotation (Andreas) shrinks to a **small
validation set** on clearly-moving, single-animal crops, plus the moving/stationary call.

Other locked decisions: project is **English**; everything must read **human-written, not AI**
(no AI-authorship traces, no banner comments / essay docstrings); **Andreas annotates** on his own
machine via the standalone package; species shown as class id, not a guessed name.

## 5. What is actually true so far (label-free; full detail in results.md, caveats in audit.md)
- **Tracking works.** It recovers genuine movement direction for clearly-moving animals (verified
  visually on real frames AND by refuting the drone-artefact hypothesis: consecutive registration
  removes ~800 px of cumulative drone motion and leaves a coherent residual, background cancels in
  the per-step difference images; honest signal/noise ~30-40x at the net level, ~3x per step - the
  earlier "~290x" was a unit error, see audit.md). The gate yields 190 "trusted", but it is lenient -
  the **defensible high-confidence core is 138** (>=8 steps, >=50 px). Caveat: an ID-switch confound
  across the whole trusted/core set, which lives almost entirely in crowded scenes (median ~7-8
  same-class animals/frame; only ~3% are single-animal tracks). The tracker has no appearance model,
  so the R/Rayleigh coherence gate is the only switch defense - not just an issue for the 18 >1000 px
  tracklets, and the visual ego-motion check used an atypical single-animal flight.
- **Single-image direction is weak.** GST is the best single-image estimator (~29 deg median axial
  error) but barely beats - and on the most common class loses to - a constant per-class-heading
  prior. Not a substitute for tracking.
- **Moving/stationary classification** (frozen DINOv2/CLIP/BioCLIP vs hand-features vs from-scratch
  CNN): foundation models edge ahead but are within noise of each other; the proxy labels are
  flight-clustered (scene leakage), so the result is weak and confounded - re-check with human labels.
- **EDA:** per-class size/intensity stats done; the boar "refined 0.37" is a segmentation failure on
  low-contrast boars, not loose boxes.
- All headline numbers were recomputed from the CSVs; no hallucinations; overstatements corrected.

## 6. Repo guide
- Interpreter (an Application Control policy blocks DLLs under Desktop\, so use this one):
  `C:/Users/rapha/AutoCode/bambi-analysis/.venv/Scripts/python.exe`. `requirements.txt` reproduces it.
- `config.py` - paths, dataset coords, class map, constants.
- `src/` - data_loader, crops, gst, bb_refinement, metrics_circular (circular stats), quality;
  registration + tracking + direction (tracking GT); blur (single-image axis); movement + deep_features
  (classification); annotation/ (label_store + annotate tool).
- `scripts/` - run_tracking[_v2], eval_blur, movement_experiment, eda, ghosting_scan,
  build_annotation_package (-> dist/bambi_annotation.zip for Andreas).
- `tests/` - 136 passing. Run: `"<interp>" -m pytest tests/ -q` from the project root.
- `output/` (gitignored) - CSVs incl. tracking_directions.csv (canonical = v2, 190), _v1 backup,
  blur_eval.csv, movement_results.csv, eda_stats.csv. `dist/` (gitignored) - the annotation package.
- `annotations/` - Andreas's `labels.csv` (1,500 crops), kept in git.

## 7. Human-label validation (DONE — see docs/validation.md)
Andreas labelled all 1,500 crops (`annotations/labels.csv`). Outcome (`scripts/validate_labels.py`):
- **Tracking direction validated:** human axis vs tracking axis ~22.7 deg median (19.1 on
  human-confirmed movers), Acc@45 0.79-0.86, vs ~50 deg random. The tracking GT holds.
- **Head-discernibility is 14 %** (27 % among movers) — confirms why we use tracking, not hand labels.
- **GST is a good orientation estimator** vs the human axis (~10.7 deg), but signed movement direction
  stays hard (GST vs tracking-movement ~29 deg).
- **Moving/stationary on real labels** (flight-disjoint): LogReg 0.62, below the 0.78 scene ceiling —
  same conclusion as the proxy. No part of the pipeline had to be redesigned.

Still open: an optional direction-regression DL arm (data-limited), then the written report.

## 8. Status
Label-free pipeline + evaluation + self-audit complete; **human labels received and the tracking
ground truth is validated** (docs/validation.md). Remaining: optional DL direction arm + the report.
