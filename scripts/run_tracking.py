"""Run the movement-direction pipeline over every flight.

Pipeline: load YOLO labels -> link detections into tracklets per flight -> for each
consecutive step register the two frames on the background and take the animal's
residual displacement -> aggregate per tracklet into a heading with confidence. Writes
output/tracking_directions.csv.

Image reads dominate the runtime, so we register each consecutive frame PAIR once and
share that transform across all tracklets that span it, and keep a small LRU of decoded
images. Expect 10-30 min for the full set.
"""
import os
import sys
import time
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd

from config import DATASET_DIR, IMG_SIZE, OUTPUT_DIR
from src.data_loader import load_annotations
from src.tracking import build_tracklets
from src.direction import residual_heading, tracklet_direction

OUT_CSV = OUTPUT_DIR / "tracking_directions.csv"
_CACHE_SIZE = 24


class ImageCache:
    """Small LRU of decoded grayscale frames keyed by (split, image_file)."""

    def __init__(self, size=_CACHE_SIZE):
        self.size = size
        self.store = OrderedDict()

    def get(self, split, image_file):
        key = (split, image_file)
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        path = os.path.join(DATASET_DIR, split, "images", image_file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        self.store[key] = img
        if len(self.store) > self.size:
            self.store.popitem(last=False)
        return img


def _frame_lookup(flight_df):
    """frame_num -> (split, image_file) for one flight (one image per frame)."""
    out = {}
    for fr, g in flight_df.groupby("frame_num"):
        r = g.iloc[0]
        out[int(fr)] = (r["split"], r["image_file"])
    return out


def process_flight(flight_df, cache, transform_cache):
    """Yield a direction record per tracklet in this flight."""
    from src.registration import estimate_transform

    frame_files = _frame_lookup(flight_df)
    records = []

    for tid, t in flight_df.groupby("tracklet_id"):
        t = t.sort_values("frame_num")
        frames = t["frame_num"].to_numpy()
        cx = t["cx"].to_numpy()
        cy = t["cy"].to_numpy()
        cls = int(t["cls"].iloc[0])
        flight_id = int(t["flight_id"].iloc[0])

        headings, residuals, inliers = [], [], []
        for k in range(1, len(frames)):
            fa, fb = int(frames[k - 1]), int(frames[k])
            key = (flight_id, fa, fb)
            if key not in transform_cache:
                ia = cache.get(*frame_files[fa])
                ib = cache.get(*frame_files[fb])
                if ia is None or ib is None:
                    transform_cache[key] = (None, 0.0)
                else:
                    transform_cache[key] = estimate_transform(ia, ib)
            M, inlier_ratio = transform_cache[key]

            h, res = residual_heading(M, (cx[k - 1], cy[k - 1]), (cx[k], cy[k]))
            headings.append(h)
            residuals.append(res)
            inliers.append(inlier_ratio)

        d = tracklet_direction(headings, residuals, inliers)
        records.append({
            "tracklet_id": int(tid),
            "flight_id": flight_id,
            "cls": cls,
            "n_steps": d.n_steps,
            "n_obs": int(len(frames)),
            "mean_dir_deg": round(d.mean_dir_deg, 2) if d.n_steps else float("nan"),
            "R": round(d.R, 4),
            "rayleigh_p": round(d.rayleigh_p, 6),
            "median_inlier": round(d.median_inlier, 4),
            "disp_px": round(d.disp_px, 2),
            "trusted": bool(d.trusted),
        })
    return records


def main():
    t0 = time.time()
    df = load_annotations()
    df = df[df.flight_id != -1].copy()
    tracked = build_tracklets(df, img_size=IMG_SIZE)

    cache = ImageCache()
    rows = []
    flight_ids = sorted(tracked["flight_id"].unique())
    for n, flight_id in enumerate(flight_ids, 1):
        fdf = tracked[tracked["flight_id"] == flight_id]
        rows.extend(process_flight(fdf, cache, transform_cache={}))
        if n % 10 == 0 or n == len(flight_ids):
            print(f"[{n}/{len(flight_ids)}] flight {flight_id} "
                  f"-> {len(rows)} tracklets, {time.time() - t0:.0f}s", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)

    trusted = res[res.trusted]
    print(f"\nwrote {OUT_CSV}")
    print(f"tracklets: {len(res)}  trusted: {len(trusted)}")
    print("trusted by class:", trusted.groupby("cls").size().to_dict())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
