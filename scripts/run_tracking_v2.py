"""Re-run the movement-direction pipeline with the improved registration and write
output/tracking_directions_v2.csv (leaving the existing v1 output untouched)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from config import IMG_SIZE, OUTPUT_DIR
from src.data_loader import load_annotations
from src.tracking import build_tracklets
from scripts.run_tracking import ImageCache, process_flight

OUT_CSV = OUTPUT_DIR / "tracking_directions_v2.csv"


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
    print(f"median per-tracklet inlier: {res.median_inlier.median():.4f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
