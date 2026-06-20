"""Exploratory data analysis over the thermal AOS wildlife dataset.

Per-class statistics the team asked for - size per species, colour/intensity, and the
effect of tightening the (oversized) manual boxes - plus flight-level counts. Box-level
stats use every detection; the refine and intensity passes load images, so they run on a
stratified sample per class. Writes output/eda_stats.csv and output/eda_*.png and prints
a short summary.

Run from the project root with the project interpreter:
    python scripts/eda.py [--refine-sample N] [--intensity-sample N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (DATASET_DIR, IMG_SIZE, OUTPUT_DIR, CLASS_NAMES, CLASS_EN,
                    SEED, FIGURE_DPI)
from src.data_loader import load_annotations
from src.crops import extract_crop, contrast_to_surround
from src.bb_refinement import refine_box

REFINE_PAD = 0.30
CLASS_ORDER = [0, 1, 2]


def _cls_label(c):
    return f"{CLASS_NAMES.get(c, c)} ({CLASS_EN.get(c, '')})"


def _read_gray(split, image_file):
    return cv2.imread(os.path.join(DATASET_DIR, split, "images", image_file),
                      cv2.IMREAD_GRAYSCALE)


def _savefig(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def box_level_stats(df):
    """Counts, longest-side px and aspect ratio per class - over every detection."""
    df = df.copy()
    df["long_px"] = np.maximum(df.w, df.h) * IMG_SIZE
    df["short_px"] = np.minimum(df.w, df.h) * IMG_SIZE
    # aspect ratio >= 1 (long/short), guard against zero-height boxes
    df["aspect"] = df["long_px"] / df["short_px"].clip(lower=1.0)
    return df


def sample_per_class(df, n, rng):
    """Up to n rows per class, sampled without replacement."""
    parts = []
    for c in CLASS_ORDER:
        sub = df[df.cls == c]
        if len(sub) > n:
            sub = sub.sample(n, random_state=int(rng.integers(1 << 31)))
        parts.append(sub)
    return pd.concat(parts).reset_index(drop=True)


def refine_pass(df, n, rng):
    """Run refine_box on a per-class sample; collect refined long side and shrink ratio.

    Original box area uses the manual w,h in pixels. Refined area is the tightened blob
    box from bb_refinement, mapped back to full-image pixels via the crop scale.
    """
    samp = sample_per_class(df, n, rng)
    rows = []
    # group by image so each frame is decoded once
    for (split, image_file), grp in samp.groupby(["split", "image_file"], sort=False):
        img = _read_gray(split, image_file)
        if img is None:
            continue
        for _, r in grp.iterrows():
            crop, _ = extract_crop(img, r.xc, r.yc, r.w, r.h, pad=REFINE_PAD,
                                   img_size=IMG_SIZE)
            if crop.size == 0 or min(crop.shape[:2]) < 8:
                continue
            rb = refine_box(crop)
            orig_w, orig_h = r.w * IMG_SIZE, r.h * IMG_SIZE
            orig_long = max(orig_w, orig_h)
            orig_area = orig_w * orig_h
            row = {"cls": int(r.cls), "orig_long_px": float(orig_long),
                   "refined_ok": bool(rb.success)}
            if rb.success:
                ref_w, ref_h = rb.x1 - rb.x0, rb.y1 - rb.y0
                row["refined_long_px"] = float(max(ref_w, ref_h))
                ref_area = float(ref_w * ref_h)
                row["shrink_ratio"] = ref_area / orig_area if orig_area > 0 else np.nan
            else:
                row["refined_long_px"] = np.nan
                row["shrink_ratio"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def intensity_pass(df, n, rng):
    """Mean/std intensity inside the manual box and contrast-to-surround, per class."""
    samp = sample_per_class(df, n, rng)
    rows = []
    for (split, image_file), grp in samp.groupby(["split", "image_file"], sort=False):
        img = _read_gray(split, image_file)
        if img is None:
            continue
        for _, r in grp.iterrows():
            x0 = int((r.xc - r.w / 2) * IMG_SIZE)
            y0 = int((r.yc - r.h / 2) * IMG_SIZE)
            x1 = int((r.xc + r.w / 2) * IMG_SIZE)
            y1 = int((r.yc + r.h / 2) * IMG_SIZE)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(IMG_SIZE, x1), min(IMG_SIZE, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            inner = img[y0:y1, x0:x1]
            rows.append({
                "cls": int(r.cls),
                "mean_int": float(inner.mean()),
                "std_int": float(inner.std()),
                "contrast": contrast_to_surround(img, r.xc, r.yc, r.w, r.h, IMG_SIZE),
            })
    return pd.DataFrame(rows)


def flight_stats(df):
    """Frames and detections per flight, and the animals-per-image distribution."""
    valid = df[df.flight_id != -1]
    per_img = df.groupby(["split", "image_file"]).size()
    per_flight = valid.groupby("flight_id").agg(
        detections=("cls", "size"),
        frames=("frame_num", "nunique"),
    )
    return per_flight, per_img


def _q(series, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    return {f"p{int(q * 100)}": float(series.quantile(q)) for q in qs}


def build_summary(box_df, refine_df, int_df):
    """Tidy one-row-per-class summary table."""
    rows = []
    for c in CLASS_ORDER:
        b = box_df[box_df.cls == c]
        rf = refine_df[refine_df.cls == c]
        it = int_df[int_df.cls == c]
        rf_ok = rf[rf.refined_ok]
        row = {
            "cls": c,
            "cls_name": CLASS_NAMES.get(c, c),
            "cls_en": CLASS_EN.get(c, ""),
            "n_boxes": int(len(b)),
            "long_px_median": float(b.long_px.median()),
            "long_px_mean": float(b.long_px.mean()),
            "long_px_p05": float(b.long_px.quantile(0.05)),
            "long_px_p95": float(b.long_px.quantile(0.95)),
            "aspect_median": float(b.aspect.median()),
            "refine_n": int(len(rf)),
            "refine_ok_frac": float(rf.refined_ok.mean()) if len(rf) else np.nan,
            "refined_long_px_median": float(rf_ok.refined_long_px.median()) if len(rf_ok) else np.nan,
            "shrink_ratio_median": float(rf_ok.shrink_ratio.median()) if len(rf_ok) else np.nan,
            "shrink_ratio_mean": float(rf_ok.shrink_ratio.mean()) if len(rf_ok) else np.nan,
            "intensity_n": int(len(it)),
            "mean_int_median": float(it.mean_int.median()) if len(it) else np.nan,
            "std_int_median": float(it.std_int.median()) if len(it) else np.nan,
            "contrast_median": float(it.contrast.median()) if len(it) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _hist_by_class(ax, df, col, value_fn=None, bins=40, title="", xlabel=""):
    for c in CLASS_ORDER:
        sub = df[df.cls == c]
        vals = value_fn(sub) if value_fn else sub[col]
        vals = np.asarray(vals)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        ax.hist(vals, bins=bins, alpha=0.5, label=_cls_label(c), density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend(fontsize=8)


def make_figures(box_df, refine_df, int_df, per_flight, per_img):
    written = []

    # class balance
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [int((box_df.cls == c).sum()) for c in CLASS_ORDER]
    ax.bar([_cls_label(c) for c in CLASS_ORDER], counts,
           color=["#d62728", "#2ca02c", "#1f77b4"])
    ax.set_ylabel("detections")
    ax.set_title("Detections per class")
    for i, v in enumerate(counts):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    written.append(_savefig(fig, "eda_class_balance.png"))

    # longest-side distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    _hist_by_class(ax, box_df, "long_px", bins=50,
                   title="Box longest side (manual boxes)", xlabel="longest side (px)")
    ax.set_xlim(0, box_df.long_px.quantile(0.99))
    written.append(_savefig(fig, "eda_size_longside.png"))

    # aspect ratio
    fig, ax = plt.subplots(figsize=(7, 4))
    _hist_by_class(ax, box_df, "aspect", bins=50,
                   title="Box aspect ratio (long/short)", xlabel="aspect ratio")
    ax.set_xlim(1, box_df.aspect.quantile(0.99))
    written.append(_savefig(fig, "eda_aspect.png"))

    # refined size + shrink ratio
    rf_ok = refine_df[refine_df.refined_ok]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _hist_by_class(axes[0], rf_ok, "refined_long_px", bins=40,
                   title="Refined longest side", xlabel="refined longest side (px)")
    _hist_by_class(axes[1], rf_ok, "shrink_ratio", bins=40,
                   title="Area shrink (refined / original)", xlabel="area ratio")
    axes[1].axvline(1.0, color="k", lw=0.8, ls="--")
    axes[1].set_xlim(0, 1.2)
    written.append(_savefig(fig, "eda_refined_size.png"))

    # intensity + contrast
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _hist_by_class(axes[0], int_df, "mean_int", bins=40,
                   title="Mean intensity in box", xlabel="mean intensity (0-255)")
    _hist_by_class(axes[1], int_df, "std_int", bins=40,
                   title="Std intensity in box", xlabel="std intensity")
    _hist_by_class(axes[2], int_df, "contrast", bins=40,
                   title="Contrast to surround", xlabel="box mean - ring mean")
    written.append(_savefig(fig, "eda_intensity.png"))

    # flight / animals-per-image
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(per_flight.detections, bins=30, color="#555")
    axes[0].set_title("Detections per flight")
    axes[0].set_xlabel("detections")
    axes[0].set_ylabel("flights")
    api = per_img.values
    bins = np.arange(0.5, api.max() + 1.5)
    axes[1].hist(api, bins=bins, color="#555")
    axes[1].set_title("Animals per image")
    axes[1].set_xlabel("detections in image")
    axes[1].set_ylabel("images")
    axes[1].set_yscale("log")
    written.append(_savefig(fig, "eda_flights.png"))

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine-sample", type=int, default=800,
                    help="crops per class for the refine pass")
    ap.add_argument("--intensity-sample", type=int, default=800,
                    help="crops per class for the intensity pass")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)

    print("loading annotations...")
    df = load_annotations()
    box_df = box_level_stats(df)
    print(f"  {len(box_df):,} detections over {df.image_file.nunique():,} images")

    print(f"refine pass ({args.refine_sample}/class)...")
    refine_df = refine_pass(box_df, args.refine_sample, rng)

    print(f"intensity pass ({args.intensity_sample}/class)...")
    int_df = intensity_pass(box_df, args.intensity_sample, rng)

    per_flight, per_img = flight_stats(df)

    summary = build_summary(box_df, refine_df, int_df)
    out_csv = OUTPUT_DIR / "eda_stats.csv"
    summary.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    figs = make_figures(box_df, refine_df, int_df, per_flight, per_img)
    for p in figs:
        print(f"wrote {p}")

    # --- text summary ---
    print("\nPER-CLASS SUMMARY")
    hdr = (f"  {'class':<22} {'n':>7} {'long_px':>8} {'refined':>8} "
           f"{'shrink':>7} {'mean_I':>7} {'contrast':>9}")
    print(hdr)
    for _, r in summary.iterrows():
        print(f"  {r.cls_name + ' (' + r.cls_en + ')':<22} {r.n_boxes:>7,} "
              f"{r.long_px_median:>8.1f} {r.refined_long_px_median:>8.1f} "
              f"{r.shrink_ratio_median:>7.2f} {r.mean_int_median:>7.1f} "
              f"{r.contrast_median:>9.1f}")
    print("  (long_px / refined = median longest side px; shrink = refined/original area)")

    print("\nANIMALS PER IMAGE")
    api = per_img
    print(f"  images with >=1 detection: {len(api):,}")
    print(f"  mean {api.mean():.2f}   median {int(api.median())}   "
          f"max {int(api.max())}   p95 {int(api.quantile(0.95))}")
    multi = int((api > 1).sum())
    print(f"  images with >1 animal: {multi:,} ({100 * multi / len(api):.1f}%)")

    print("\nFLIGHTS")
    print(f"  flights: {len(per_flight)}   "
          f"detections/flight median {int(per_flight.detections.median())}   "
          f"frames/flight median {int(per_flight.frames.median())}")


if __name__ == "__main__":
    main()
