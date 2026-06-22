# CLAUDE.md — read this first

Computer-Vision course project (FH Hagenberg, team of 2). Task: **estimate the movement direction of
wildlife from thermal Airborne Optical Sectioning (AOS) light-field drone imagery.**

## START HERE
1. **`docs/PROJECT.md`** — the single ground-truth doc (task, decoded proposal, dataset, decisions,
   verified state, repo guide). Read it before doing anything.
2. `docs/results.md` (numbers) · `docs/audit.md` (what is overstated and why — critical self-review) ·
   `docs/plans/2026-06-19-movement-direction-design.md` (the plan) · `docs/references.md` (literature).
3. `docs/lernhilfe-praesentation.md` (German study/prez doc) · `docs/BAMBI_Praesentation.pptx` / `.pdf`.

## The core idea (one paragraph)
There is **no ground truth for direction** (the head is rarely visible on the blurry warm blobs). So the
direction ground truth comes from **TRACKING**: register consecutive frames to cancel drone ego-motion,
and the animal's residual displacement is its real heading. On top of that we **compare** classical CV,
a from-scratch CNN, and frozen foundation models, with honest, leakage-free evaluation. Human labels
(Andreas) are only a small **validation** set.

## Honesty pillars (never state otherwise)
1. **No detector/YOLO was trained** — the boxes came ready in YOLO format with the dataset.
2. **Segmentation is classical** (Otsu + morphology + connected components + image moments), not a neural net.
3. **Tracking is a simple greedy nearest-centroid associator** (no Kalman/SORT), chosen deliberately.
4. **Species mapping is unconfirmed** — the dataset stores only class ids 0/1/2; Rotwild/Rehwild/Schwarzwild
   is an unverified assumption.
5. **Direction truth = tracking**; human labels only validate.

## How to work here (user preferences)
- Everything in the repo is **English** and must read **human-written** — **no AI-authorship traces**
  in code, comments, or commit messages (no `Co-Authored-By`, no "Generated with…").
- Be rigorous and self-critical; reproduce numbers from the CSVs, don't assert.
- Don't blindly trust the old `bambi-analysis` repo — re-derive and validate.

## Verify the claims (no raw dataset needed)
`python scripts/verify_claims.py` re-derives the label-free headline numbers (138 core, the trusted
gate, GST 29.1°, the per-class baselines, the movement table, …) directly from the committed
`output/*.csv`, and re-checks the human-label numbers (linchpin 22.7°, GST-vs-human 10.7°, head 14 %,
the real-label movement table, the motion-granularity decision) against the committed
`output/label_validation_summary.csv` — needs only pandas + numpy. All checks pass. The *from-scratch*
human-label re-derivation (which rebuilds that summary from `annotations/labels.csv` + the raw images)
is `scripts/validate_labels.py`, and DOES need the raw dataset — see below.

## Repo guide
- `src/` library code · `scripts/` runnable pipeline steps · `tests/` (run: `python -m pytest tests/ -q`,
  136 passing) · `config.py` paths/constants · `annotations/` (Andreas's `labels.csv`, committed).
- `docs/` everything above. `requirements.txt` reproduces the Python env.

## Not in git (set up per machine)
- **Dataset** (~12k thermal images, too big for git): Roboflow `bambi-overview/bambi-alfs-20250520-upload04-sdakr`
  v2 (see `config.py`). Re-download with `ROBOFLOW_API_KEY` in `.env`, or copy the local `data/` folder over.
- **Python env**: `python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt`
  (add `torch` cu121 for the DL parts). Note: on the original machine an Application-Control policy blocked
  compiled DLLs under `Desktop\`, so a dedicated interpreter was used; on a normal machine a fresh venv works.
- The small **result CSVs and figures in `output/` ARE committed** (so the claims are checkable without
  the raw images); only `output/crops/`, `output/deep_feat_cache/` and `dist/` are gitignored.

## Pending (next work)
Andreas's `labels.csv` (1,500 crops) arrived and the validation is **done** — see `docs/validation.md`:
the tracking direction agrees with human perception (~22.7 deg median axis error), head-discernibility
is only 14 %, and moving/stationary on real labels matches the proxy. Nothing in the pipeline had to be
redesigned. Remaining: an optional direction-regression DL arm (flight-disjoint, data-limited), then the
written report.
