"""Build tracklets per flight by linking detections across consecutive frames.

Scenes are sparse (a few animals per frame), so greedy nearest-centroid association
under a distance gate is enough. Same class only; the gate is generous and scales with
the frame gap because sampling is irregular (gaps run from 1 to ~60 frames). Centroids
are in full-res pixels. Each tracklet gets a global id.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import IMG_SIZE


def associate(prev: pd.DataFrame, curr: pd.DataFrame, gate_px: float):
    """Greedy nearest-centroid matches between two frames, same class, under the gate.

    prev/curr need columns cx, cy, cls. Returns list of (prev_pos, curr_pos) integer
    positions (0..len-1), smallest distance first, each detection used at most once.
    """
    p_cx, p_cy, p_cls = prev["cx"].to_numpy(), prev["cy"].to_numpy(), prev["cls"].to_numpy()
    c_cx, c_cy, c_cls = curr["cx"].to_numpy(), curr["cy"].to_numpy(), curr["cls"].to_numpy()

    pairs = []
    for i in range(len(prev)):
        for j in range(len(curr)):
            if p_cls[i] != c_cls[j]:
                continue
            d = float(np.hypot(p_cx[i] - c_cx[j], p_cy[i] - c_cy[j]))
            if d <= gate_px:
                pairs.append((d, i, j))

    pairs.sort(key=lambda t: t[0])
    used_prev, used_curr, out = set(), set(), []
    for _, i, j in pairs:
        if i in used_prev or j in used_curr:
            continue
        used_prev.add(i)
        used_curr.add(j)
        out.append((i, j))
    return out


def build_tracklets(df: pd.DataFrame, gate_base_px: float = 120.0,
                    gate_per_gap_px: float = 12.0, img_size: int = IMG_SIZE) -> pd.DataFrame:
    """Assign a global ``tracklet_id`` to every detection.

    Expects columns flight_id, frame_num, cls, xc, yc (normalized centers). Adds cx, cy
    (pixel centers) and tracklet_id. A detection with no match in the previous frame
    starts a new tracklet.
    """
    out = df.copy().reset_index(drop=True)
    out["cx"] = out["xc"] * img_size
    out["cy"] = out["yc"] * img_size
    out["tracklet_id"] = -1
    next_id = 0

    for flight_id, fdf in out.groupby("flight_id", sort=True):
        if flight_id == -1:
            continue
        frames = sorted(fdf["frame_num"].unique())
        if not frames:
            continue

        by_frame = {fr: fdf[fdf["frame_num"] == fr] for fr in frames}
        for idx in by_frame[frames[0]].index:
            out.at[idx, "tracklet_id"] = next_id
            next_id += 1

        for k in range(1, len(frames)):
            gap = frames[k] - frames[k - 1]
            gate = gate_base_px + gate_per_gap_px * gap
            prev_rows = by_frame[frames[k - 1]]
            curr_rows = by_frame[frames[k]]

            matched_curr = set()
            for pi, ci in associate(prev_rows, curr_rows, gate):
                out.at[curr_rows.index[ci], "tracklet_id"] = out.at[prev_rows.index[pi], "tracklet_id"]
                matched_curr.add(ci)

            for ci in range(len(curr_rows)):
                if ci not in matched_curr:
                    out.at[curr_rows.index[ci], "tracklet_id"] = next_id
                    next_id += 1

    return out
