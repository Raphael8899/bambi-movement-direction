"""Build the standalone annotation package that Andreas downloads and runs.

It selects a stratified set of crops, writes them plus a manifest into a folder
together with the (self-contained) tool, and zips the whole thing. The only thing
the recipient needs is Python 3 with Pillow.

Selection: per species we keep crops where the animal is clearly warmer than its
surroundings, then sample evenly across three elongation bins (compact .. streaked)
so the annotator sees both stationary-looking and motion-blurred animals. The final
manifest order is shuffled, so stopping part-way still gives a representative sample.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd

from config import DATASET_DIR, IMG_SIZE, CLASS_NAMES, PROJECT_ROOT, SEED
from src.data_loader import load_annotations
from src.crops import extract_crop, contrast_to_surround, crop_coherence

PER_SPECIES = {"Rotwild": 500, "Rehwild": 400, "Schwarzwild": 300}
MIN_CONTRAST = 6.0
SIZE_RANGE = (50, 150)
PAD = 0.30

PKG_DIR = PROJECT_ROOT / "dist" / "bambi_annotation"
CROPS_DIR = PKG_DIR / "crops"
TOOL_SRC = PROJECT_ROOT / "src" / "annotation"


def select_for_species(df_species, n, rng):
    """Read candidates, keep the visible ones, return up to n stratified by elongation."""
    kept = []
    cache = {}
    order = rng.permutation(len(df_species))
    max_scan = int(2.5 * n) + 50
    for k in order[:max_scan]:
        r = df_species.iloc[int(k)]
        ip = os.path.join(DATASET_DIR, r.split, "images", r.image_file)
        im = cache.get(ip)
        if im is None:
            im = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            if im is None:
                continue
            if len(cache) < 48:
                cache[ip] = im
        if contrast_to_surround(im, r.xc, r.yc, r.w, r.h, IMG_SIZE) < MIN_CONTRAST:
            continue
        crop, _ = extract_crop(im, r.xc, r.yc, r.w, r.h, pad=PAD, img_size=IMG_SIZE)
        if crop.size == 0 or min(crop.shape[:2]) < 16:
            continue
        kept.append((r, crop_coherence(crop), crop))
        if len(kept) >= int(1.4 * n):
            break

    if not kept:
        return []
    cohs = np.array([c for _, c, _ in kept])
    q1, q2 = np.quantile(cohs, [1 / 3, 2 / 3])
    bins = {0: [], 1: [], 2: []}
    for item in kept:
        c = item[1]
        bins[0 if c <= q1 else (1 if c <= q2 else 2)].append(item)
    picked = []
    per_bin = n // 3 + 1
    for b in (0, 1, 2):
        idx = rng.permutation(len(bins[b]))[:per_bin]
        picked.extend(bins[b][i] for i in idx)
    rng.shuffle(picked)
    return picked[:n]


def main():
    rng = np.random.default_rng(SEED)
    df = load_annotations()
    df = df[df.long_px.between(*SIZE_RANGE)].copy()

    if CROPS_DIR.exists():
        shutil.rmtree(PKG_DIR)
    CROPS_DIR.mkdir(parents=True)

    rows = []
    for cls, name in CLASS_NAMES.items():
        n = PER_SPECIES[name]
        picked = select_for_species(df[df.cls == cls].reset_index(drop=True), n, rng)
        for r, _, crop in picked:
            crop_id = f"{r.split}_{r.flight_id}_{r.frame_num}_{r.line_num}"
            fname = f"crops/{crop_id}.png"
            cv2.imwrite(str(PKG_DIR / fname), crop)
            rows.append({"crop_id": crop_id, "file": fname, "species": name,
                         "flight_id": int(r.flight_id), "frame_num": int(r.frame_num),
                         "orig_long_px": round(float(r.long_px), 1)})
        print(f"{name}: {len(picked)} crops")

    manifest = pd.DataFrame(rows).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    manifest.to_csv(PKG_DIR / "manifest.csv", index=False)

    shutil.copy2(TOOL_SRC / "annotate.py", PKG_DIR / "annotate.py")
    shutil.copy2(TOOL_SRC / "label_store.py", PKG_DIR / "label_store.py")
    (PKG_DIR / "requirements.txt").write_text("pillow\n", encoding="utf-8")
    (PKG_DIR / "run.bat").write_text(
        "@echo off\r\npython annotate.py\r\npause\r\n", encoding="utf-8")
    (PKG_DIR / "README.md").write_text(_README, encoding="utf-8")

    archive = shutil.make_archive(str(PROJECT_ROOT / "dist" / "bambi_annotation"),
                                  "zip", root_dir=PKG_DIR.parent, base_dir="bambi_annotation")
    total = len(manifest)
    print(f"\n{total} crops total")
    print("package:", PKG_DIR)
    print("zip:", archive)


_README = """# BAMBI annotation

You are labelling small thermal images of single wild animals seen from a drone.
For each crop, set two things and move on. It saves after every crop and resumes
where you left off, so you can stop and continue any time.

## Setup (once)
1. Install Python 3 (python.org). On Windows tick "Add python.exe to PATH".
2. In this folder run:  `pip install -r requirements.txt`

## Start
Double-click `run.bat` (Windows), or from this folder run:

    python annotate.py

(an output file `labels.csv` appears in this folder)

## What to do per crop
1. Is the animal sharp or smeared?
   - `s` stationary (crisp blob)   `d` moving (smeared / streaked)   `u` unsure
2. Which way does the HEAD point? The number keys mirror a compass (up = top of image):

       7 NW    8 N     9 NE
       4 W     5 axis  6 E
       1 SW    2 S     3 SE

   - `1`-`9` (not 5): head points that way
   - `5`: you can see the body line but not which end is the head
   - `0`: nothing usable in this crop

Then `Enter` (or `Space`) for the next one. `Backspace` goes back to fix a crop,
`Esc` saves and quits.

## Notes
- Be honest. If you can't tell the head from the tail, press `5`. Don't guess.
- Red deer are usually the clearest; roe deer and boar are often axis-only.
- Aim for roughly 4-5 hours. The order is random, so whatever you finish is fine
  - just get through as many as you can.

When done, send back `labels.csv`.
"""

if __name__ == "__main__":
    main()
