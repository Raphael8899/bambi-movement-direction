"""Build the standalone annotation package used for manual labelling.

We only keep crops with exactly ONE animal in view (no herds, no neighbours), where the
animal is clearly warmer than the background. Each saved crop is contrast-stretched and
has the target animal drawn in green so it's obvious what to judge. We spread the
selection across species and across an elongation score (a rough stationary/moving mix)
and shuffle the order, so labelling any prefix still gives a representative sample.

Sized for ~5 hours: ~1500 crops. The result is a self-contained folder + zip that runs
with just Python 3 and Pillow.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd

from config import DATASET_DIR, IMG_SIZE, CLASS_NAMES, PROJECT_ROOT, SEED
from src.data_loader import load_annotations, yolo_to_box
from src.crops import contrast_to_surround, crop_coherence

PER_SPECIES = {"Rotwild": 600, "Rehwild": 500, "Schwarzwild": 400}
MIN_CONTRAST = 12.0
SIZE_RANGE = (50, 150)
PAD = 0.30

PKG = PROJECT_ROOT / "dist" / "bambi_annotation"
TOOL = PROJECT_ROOT / "src" / "annotation"


def region(xc, yc, w, h):
    cx, cy = xc * IMG_SIZE, yc * IMG_SIZE
    half = max(w, h) * IMG_SIZE * (0.5 + PAD)
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(IMG_SIZE, int(cx + half)), min(IMG_SIZE, int(cy + half))
    return x0, y0, x1, y1


def alone_in_region(target, others, reg):
    """True if no OTHER box overlaps the crop region (single animal in view)."""
    x0, y0, x1, y1 = reg
    for o in others:
        if o.line_num == target.line_num:
            continue
        bx0, by0, bx1, by1 = yolo_to_box(o.xc, o.yc, o.w, o.h, IMG_SIZE)
        if not (bx1 < x0 or bx0 > x1 or by1 < y0 or by0 > y1):
            return False
    return True


def select(df_species, by_image, n, rng):
    kept = []
    cache = {}
    order = rng.permutation(len(df_species))
    for k in order:
        if len(kept) >= int(1.4 * n):
            break
        r = df_species.iloc[int(k)]
        reg = region(r.xc, r.yc, r.w, r.h)
        siblings = list(by_image[r.image_file].itertuples(index=False))
        if not alone_in_region(r, siblings, reg):
            continue
        ip = os.path.join(DATASET_DIR, r.split, "images", r.image_file)
        im = cache.get(ip)
        if im is None:
            im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            if len(cache) < 64:
                cache[ip] = im
        if contrast_to_surround(im, r.xc, r.yc, r.w, r.h, IMG_SIZE) < MIN_CONTRAST:
            continue
        x0, y0, x1, y1 = reg
        crop = im[y0:y1, x0:x1]
        if crop.size == 0 or min(crop.shape[:2]) < 24:
            continue
        kept.append((r, crop_coherence(crop), crop, (x0, y0)))

    if not kept:
        return []
    cohs = np.array([c for _, c, _, _ in kept])
    q1, q2 = np.quantile(cohs, [1 / 3, 2 / 3])
    bins = {0: [], 1: [], 2: []}
    for it in kept:
        c = it[1]
        bins[0 if c <= q1 else (1 if c <= q2 else 2)].append(it)
    out = []
    per_bin = n // 3 + 1
    for b in (0, 1, 2):
        idx = rng.permutation(len(bins[b]))[:per_bin]
        out.extend(bins[b][i] for i in idx)
    rng.shuffle(out)
    return out[:n]


def render(crop, target, origin):
    """Contrast-stretch and draw the green target box."""
    lo, hi = np.percentile(crop, [2, 99])
    cs = np.clip((crop.astype(float) - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(cs, cv2.COLOR_GRAY2BGR)
    x0, y0 = origin
    bx0, by0, bx1, by1 = yolo_to_box(target.xc, target.yc, target.w, target.h, IMG_SIZE)
    cv2.rectangle(vis, (int(bx0 - x0), int(by0 - y0)), (int(bx1 - x0), int(by1 - y0)),
                  (0, 255, 0), 2)
    return vis


def main():
    rng = np.random.default_rng(SEED)
    df = load_annotations()
    by_image = {f: g for f, g in df.groupby("image_file")}
    df = df[df.long_px.between(*SIZE_RANGE)].copy()

    if PKG.exists():
        shutil.rmtree(PKG)
    (PKG / "crops").mkdir(parents=True)

    rows = []
    for cls, name in CLASS_NAMES.items():
        picked = select(df[df.cls == cls].reset_index(drop=True), by_image, PER_SPECIES[name], rng)
        for r, _, crop, origin in picked:
            crop_id = f"{r.split}_{r.flight_id}_{r.frame_num}_{r.line_num}"
            cv2.imwrite(str(PKG / "crops" / f"{crop_id}.png"), render(crop, r, origin))
            # store the dataset CLASS ID (0/1/2), not a species name: the name->id mapping
            # is unverified and not stored in the dataset, so we persist the class id.
            rows.append({"crop_id": crop_id, "file": f"crops/{crop_id}.png", "class_id": str(cls),
                         "flight_id": int(r.flight_id), "frame_num": int(r.frame_num),
                         "orig_long_px": round(float(r.long_px), 1)})
        print(f"{name}: {len(picked)}")

    manifest = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    manifest.to_csv(PKG / "manifest.csv", index=False)

    shutil.copy2(TOOL / "annotate.py", PKG / "annotate.py")
    shutil.copy2(TOOL / "label_store.py", PKG / "label_store.py")
    (PKG / "requirements.txt").write_text("pillow\n", encoding="utf-8")
    (PKG / "run.bat").write_text("@echo off\r\npython annotate.py\r\npause\r\n", encoding="utf-8")
    (PKG / "README.md").write_text(
        (PROJECT_ROOT / "scripts" / "annotation_readme.md").read_text(encoding="utf-8"),
        encoding="utf-8")

    archive = shutil.make_archive(str(PROJECT_ROOT / "dist" / "bambi_annotation"),
                                  "zip", root_dir=PKG.parent, base_dir="bambi_annotation")
    print(f"\n{len(manifest)} crops total\nzip: {archive}")


if __name__ == "__main__":
    main()
