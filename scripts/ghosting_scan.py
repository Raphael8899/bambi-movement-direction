"""Run the ghosting heuristic over a sample of real frames and eyeball the worst ones.

Scores a random sample of dataset images with src.quality.ghosting_score, reports the
score distribution and the fraction flagged at a threshold, and saves a montage of the
highest-scoring frames to output/ghosting_examples.png. The score is a triage heuristic,
not a validated detector - the montage is there so the flagged frames can be checked by
eye.

Run from the project root with the project interpreter:
    python scripts/ghosting_scan.py [--sample N] [--thr T]
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from config import DATASET_DIR, SPLITS, OUTPUT_DIR, SEED
from src.quality import ghosting_score

THUMB = 256


def list_images():
    paths = []
    for s in SPLITS:
        paths += glob.glob(os.path.join(DATASET_DIR, s, "images", "*.jpg"))
    return sorted(paths)


def montage(paths_scores, cols=4):
    """Grid of thumbnails labelled with their score."""
    n = len(paths_scores)
    rows = (n + cols - 1) // cols
    canvas = np.zeros((rows * THUMB, cols * THUMB), np.uint8)
    for i, (p, sc) in enumerate(paths_scores):
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        th = cv2.resize(im, (THUMB, THUMB))
        cv2.putText(th, f"{sc:.2f}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    255, 2, cv2.LINE_AA)
        r, c = divmod(i, cols)
        canvas[r * THUMB:(r + 1) * THUMB, c * THUMB:(c + 1) * THUMB] = th
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=1500, help="frames to score")
    ap.add_argument("--thr", type=float, default=1.5, help="flag threshold")
    ap.add_argument("--top", type=int, default=12, help="frames in the montage")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    paths = list_images()
    if len(paths) > args.sample:
        idx = rng.choice(len(paths), size=args.sample, replace=False)
        paths = [paths[i] for i in idx]
    print(f"scoring {len(paths)} frames...")

    scored = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        scored.append((p, ghosting_score(im)))

    scores = np.array([s for _, s in scored])
    scored.sort(key=lambda t: t[1], reverse=True)

    flagged = int((scores >= args.thr).sum())
    print("\nGHOSTING SCORE DISTRIBUTION")
    for q in (0.5, 0.9, 0.95, 0.99):
        print(f"  p{int(q * 100):>2}: {np.quantile(scores, q):.3f}")
    print(f"  max: {scores.max():.3f}   mean: {scores.mean():.3f}")
    print(f"\nflagged at thr={args.thr}: {flagged}/{len(scores)} "
          f"({100 * flagged / len(scores):.1f}%)")

    top = scored[:args.top]
    canvas = montage([(p, s) for p, s in top])
    out = OUTPUT_DIR / "ghosting_examples.png"
    cv2.imwrite(str(out), canvas)
    print(f"wrote {out}  (top {len(top)} by score)")
    print("  inspect the montage - this is a heuristic, confirm the flagged frames by eye")


if __name__ == "__main__":
    main()
