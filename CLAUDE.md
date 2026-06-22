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
`python scripts/verify_claims.py` re-derives every headline number (138 core, GST 29.1°, the movement
table, EDA stats, …) from the committed `output/*.csv` and checks each against the documented value —
needs only pandas + numpy. All checks currently pass. This confirms the docs match the data without the
1.9 GB image set. (Image-level / linchpin re-checks DO need the raw dataset — see below.)

## Repo guide
- `src/` library code · `scripts/` runnable pipeline steps · `tests/` (run: `python -m pytest tests/ -q`,
  128 passing) · `config.py` paths/constants · `annotations/` (where Andreas's `labels.csv` will go).
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
When Andreas's `labels.csv` arrives: validate the tracking direction vs human perception (+ head-discernibility
rate), re-run moving/stationary and direction evaluation on real labels, add a direction-regression DL arm
(flight-disjoint), then the written report.
