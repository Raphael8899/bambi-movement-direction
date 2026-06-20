"""Evaluate single-image motion-blur axis estimators against the tracking ground truth.

For each trusted (moving) tracklet we take a few member frames around the middle,
crop the animal, and run every axis estimator: GST, moment axis (bb_refinement),
gradient/spectrum/cepstrum blur axes. The ground-truth axis is the tracklet heading
reduced to [0,180). We pool axial errors per crop and per tracklet, compare against a
random and a constant-axis baseline, and repeat on a set of stationary animals to see
whether movers actually agree better. Writes output/blur_eval.csv.

Run from the project root with the project interpreter:
    python scripts/eval_blur.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd

from config import DATASET_DIR, IMG_SIZE, OUTPUT_DIR, SEED, CLASS_NAMES
from src.data_loader import load_annotations
from src.tracking import build_tracklets
from src.crops import extract_crop
from src.gst import gst_orientation
from src.bb_refinement import refine_box
from src.blur import blur_axis_gradient, blur_axis_spectrum, blur_axis_cepstrum
from src.metrics_circular import circ_dist, circ_mean, rayleigh_p, acc_at, resultant_length

OUT_CSV = OUTPUT_DIR / "blur_eval.csv"
N_PER_TRACKLET = 5          # representative member frames around the middle
PAD = 0.3
STATIONARY_DISP_PX = 20.0   # control set: tracklets that barely move
STATIONARY_MIN_OBS = 5

METHODS = ["gst", "moment", "grad", "spectrum", "cepstrum"]


def _middle_members(members: pd.DataFrame, k: int) -> pd.DataFrame:
    """Up to k members centred on the middle of the tracklet."""
    n = len(members)
    if n <= k:
        return members
    lo = (n - k) // 2
    return members.iloc[lo:lo + k]


def _estimate_all(crop_gray: np.ndarray) -> dict:
    """Run every estimator on one crop -> {method: (axis_deg, confidence)}."""
    out = {}
    ax, coh = gst_orientation(crop_gray)
    out["gst"] = (ax, coh)

    rb = refine_box(crop_gray)
    out["moment"] = (rb.axis_deg, rb.eccentricity) if rb.success else (np.nan, 0.0)

    out["grad"] = blur_axis_gradient(crop_gray)
    out["spectrum"] = blur_axis_spectrum(crop_gray)
    cep_ax, cep_conf, _ = blur_axis_cepstrum(crop_gray)
    out["cepstrum"] = (cep_ax, cep_conf)
    return out


def _eval_tracklets(tracklets, tracked, gt_axis_by_tid, cache, group_label):
    """Per-crop prediction rows for a set of (tracklet_id, flight_id, cls)."""
    rows = []
    for tid, flight_id, cls in tracklets:
        members = tracked[tracked.tracklet_id == tid].sort_values("frame_num")
        gt_axis = gt_axis_by_tid[tid]
        for _, m in _middle_members(members, N_PER_TRACKLET).iterrows():
            img = cache.get(m["split"], m["image_file"])
            if img is None:
                continue
            crop, _ = extract_crop(img, m.xc, m.yc, m.w, m.h, pad=PAD, img_size=IMG_SIZE)
            if crop.size == 0 or min(crop.shape[:2]) < 12:
                continue
            est = _estimate_all(crop)
            row = {
                "group": group_label, "tracklet_id": tid, "flight_id": flight_id,
                "cls": cls, "frame_num": int(m.frame_num), "gt_axis": gt_axis,
                "long_px": float(m.long_px),
            }
            for meth in METHODS:
                ax, conf = est[meth]
                row[f"{meth}_axis"] = ax
                row[f"{meth}_conf"] = conf
                row[f"{meth}_err"] = (circ_dist(ax, gt_axis, period=180)
                                      if np.isfinite(ax) else np.nan)
            rows.append(row)
    return rows


class ImageCache:
    """Small LRU of decoded grayscale frames."""

    def __init__(self, size=48):
        from collections import OrderedDict
        self.size = size
        self.store = OrderedDict()

    def get(self, split, image_file):
        key = (split, image_file)
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        img = cv2.imread(os.path.join(DATASET_DIR, split, "images", image_file),
                         cv2.IMREAD_GRAYSCALE)
        self.store[key] = img
        if len(self.store) > self.size:
            self.store.popitem(last=False)
        return img


def _per_tracklet_axis(group_df: pd.DataFrame, method: str):
    """Aggregate a tracklet's crop predictions: pick the highest-confidence axis."""
    sub = group_df.dropna(subset=[f"{method}_axis"])
    if sub.empty:
        return np.nan
    return float(sub.loc[sub[f"{method}_conf"].idxmax(), f"{method}_axis"])


def _metrics(errs: np.ndarray) -> dict:
    errs = errs[np.isfinite(errs)]
    if errs.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "acc22": np.nan, "acc45": np.nan}
    return {
        "n": int(errs.size),
        "mean": float(np.mean(errs)),
        "median": float(np.median(errs)),
        "acc22": float(np.mean(errs <= 22.5)),
        "acc45": float(np.mean(errs <= 45.0)),
    }


def _print_table(title, table):
    print(f"\n{title}")
    print(f"  {'method':<10} {'n':>5} {'mean':>7} {'median':>7} {'Acc@22':>7} {'Acc@45':>7}")
    for name, m in table:
        if m["n"] == 0:
            print(f"  {name:<10} {0:>5}      -       -       -       -")
            continue
        print(f"  {name:<10} {m['n']:>5} {m['mean']:>7.1f} {m['median']:>7.1f} "
              f"{m['acc22']:>7.2f} {m['acc45']:>7.2f}")


def main():
    rng = np.random.default_rng(SEED)

    print("loading annotations and rebuilding tracklets...")
    df = load_annotations()
    df = df[df.flight_id != -1].copy()
    tracked = build_tracklets(df, img_size=IMG_SIZE)

    td = pd.read_csv(OUTPUT_DIR / "tracking_directions.csv")
    movers = td[td.trusted == True].copy()                       # noqa: E712
    movers["gt_axis"] = movers["mean_dir_deg"] % 180.0
    gt_by_tid = dict(zip(movers.tracklet_id, movers.gt_axis))

    mover_keys = [(int(r.tracklet_id), int(r.flight_id), int(r.cls))
                  for _, r in movers.iterrows()]

    # Control set: tracklets that barely move (small displacement, enough observations).
    stat = td[(td.trusted == False) & (td.n_obs >= STATIONARY_MIN_OBS)  # noqa: E712
              & (td.disp_px < STATIONARY_DISP_PX) & td.mean_dir_deg.notna()].copy()
    stat["gt_axis"] = stat["mean_dir_deg"] % 180.0
    for _, r in stat.iterrows():
        gt_by_tid[int(r.tracklet_id)] = float(r.gt_axis)
    stat_keys = [(int(r.tracklet_id), int(r.flight_id), int(r.cls))
                 for _, r in stat.iterrows()]
    print(f"movers: {len(mover_keys)}   stationary controls: {len(stat_keys)}")

    cache = ImageCache()
    print("evaluating mover crops...")
    rows = _eval_tracklets(mover_keys, tracked, gt_by_tid, cache, "mover")
    print(f"  {len(rows)} mover crops")
    print("evaluating stationary crops...")
    rows += _eval_tracklets(stat_keys, tracked, gt_by_tid, cache, "stationary")

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(res)} rows)")

    movers_df = res[res.group == "mover"]
    gt_all = movers_df["gt_axis"].to_numpy()

    # --- per-crop, pooled over all mover crops ---
    table = []
    for meth in METHODS:
        table.append((meth, _metrics(movers_df[f"{meth}_err"].to_numpy())))

    # Baselines on the same mover crops.
    rand_axis = rng.uniform(0, 180, size=len(gt_all))
    rand_err = circ_dist(rand_axis, gt_all, period=180)
    table.append(("random", _metrics(np.asarray(rand_err))))

    p = rayleigh_p(gt_all, period=180)
    const_axis = circ_mean(gt_all, period=180)
    const_err = circ_dist(np.full_like(gt_all, const_axis), gt_all, period=180)
    table.append(("constant", _metrics(np.asarray(const_err))))

    _print_table("PER-CROP axial error, movers (pooled)", table)
    print(f"  [constant = global mean axis {const_axis:.1f} deg; "
          f"Rayleigh p={p:.3g}, R={resultant_length(gt_all, period=180):.3f}]")

    # --- per-tracklet aggregate (highest-confidence crop per tracklet) ---
    tt = []
    for meth in METHODS:
        errs = []
        for tid, g in movers_df.groupby("tracklet_id"):
            ax = _per_tracklet_axis(g, meth)
            if np.isfinite(ax):
                errs.append(circ_dist(ax, gt_by_tid[tid], period=180))
        tt.append((meth, _metrics(np.asarray(errs))))
    _print_table("PER-TRACKLET axial error, movers (best-confidence crop)", tt)

    # --- per class, per-crop ---
    for cls in sorted(movers_df.cls.unique()):
        cdf = movers_df[movers_df.cls == cls]
        ct = [(meth, _metrics(cdf[f"{meth}_err"].to_numpy())) for meth in METHODS]
        cg = cdf["gt_axis"].to_numpy()
        ct.append(("constant", _metrics(np.asarray(
            circ_dist(np.full_like(cg, circ_mean(cg, period=180)), cg, period=180)))))
        _print_table(f"PER-CROP axial error, class {cls} ({CLASS_NAMES.get(cls, cls)}), "
                     f"n_tracklets={cdf.tracklet_id.nunique()}", ct)

    # --- movers vs stationary contrast (per-crop) ---
    stat_df = res[res.group == "stationary"]
    print("\nMOVERS vs STATIONARY (per-crop median axial error)")
    print(f"  {'method':<10} {'movers':>8} {'stationary':>11} {'delta':>7}")
    for meth in METHODS:
        mm = _metrics(movers_df[f"{meth}_err"].to_numpy())
        sm = _metrics(stat_df[f"{meth}_err"].to_numpy())
        if mm["n"] and sm["n"]:
            print(f"  {meth:<10} {mm['median']:>8.1f} {sm['median']:>11.1f} "
                  f"{sm['median'] - mm['median']:>+7.1f}")
    print("  (positive delta = movers agree better than stationary, as expected "
          "if blur carries direction)")


if __name__ == "__main__":
    main()
