"""Validate the tracking-derived direction ground truth against Andreas's human labels.

This is the final, honest validation: does an independent human agree with the geometry-based
tracking direction, and how good are the single-image estimators against a human reference?

Convention note: the annotation tool stores angles as 0=up, clockwise; the tracking heading is
atan2(dy, dx) with y down (0=east). They differ by 90 deg -> we map tool -> tracking with -90.
Axis comparisons are mod 180 (no head/tail); full headings mod 360.

Reads annotations/labels.csv, output/tracking_directions.csv, and the dataset; writes
output/label_validation.csv and prints a summary.

    python scripts/validate_labels.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATASET_DIR, IMG_SIZE, OUTPUT_DIR, ANNOTATIONS_DIR
from src.crops import extract_crop
from src.data_loader import load_annotations
from src.gst import gst_orientation
from src.movement import hand_features
from src.tracking import build_tracklets

MOVING = ("slight", "moderate", "strong")


def circ_dist(a, b, p):
    a = np.asarray(a, float); b = np.asarray(b, float)
    d = np.abs((a - b) % p)
    return np.minimum(d, p - d)


def section(t):
    print("\n" + "=" * 66 + "\n" + t + "\n" + "=" * 66)


def main():
    labels_path = os.path.join(ANNOTATIONS_DIR, "labels.csv")
    L = pd.read_csv(labels_path)
    parts = L.crop_id.str.split("_", expand=True)
    L["split"] = parts[0]
    L["flight_id"] = parts[1].astype(int)
    L["frame_num"] = parts[2].astype(int)
    L["line_num"] = parts[3].astype(int)
    dc = L.direction_class
    L["has_axis"] = dc >= 10            # a line (axis or head) was given
    L["has_head"] = dc >= 20            # an arrowhead (full heading) was given
    L["moving_human"] = L.motion_state.isin(MOVING)
    # tool (0=up, CW) -> tracking (0=east, CW): subtract 90
    L["human_axis"] = (L.direction_deg - 90) % 180
    L["human_full"] = (L.direction_deg - 90) % 360

    section("1. COVERAGE")
    print("labels:", len(L), " motion:", L.motion_state.value_counts().to_dict())
    print(f"direction: none={int((dc == -2).sum())}  axis-only={int(((dc >= 10) & (dc < 20)).sum())}"
          f"  head={int(L.has_head.sum())}")

    # link each labelled crop to its tracklet + tracking direction
    df = load_annotations()
    df = df[df.flight_id != -1].copy()
    tk = build_tracklets(df, img_size=IMG_SIZE)
    scc = tk.groupby(["flight_id", "frame_num", "cls"]).size().rename("scc").reset_index()
    tk = tk.merge(scc, on=["flight_id", "frame_num", "cls"])
    m = L.merge(tk[["split", "flight_id", "frame_num", "line_num", "tracklet_id",
                    "cls", "xc", "yc", "w", "h", "image_file", "scc"]],
                on=["split", "flight_id", "frame_num", "line_num"], how="left")
    td = pd.read_csv(OUTPUT_DIR / "tracking_directions.csv")
    m = m.merge(td[["tracklet_id", "trusted", "mean_dir_deg", "disp_px", "R", "n_steps"]],
                on="tracklet_id", how="left")
    m["trk_axis"] = m.mean_dir_deg % 180
    m["trk_full"] = m.mean_dir_deg % 360
    print(f"linked to a tracklet: {m.tracklet_id.notna().sum()}/{len(m)}")

    # GST (single-image estimator) on every labelled crop, in one image pass
    gst_ax = np.full(len(m), np.nan)
    hand = {}
    idx_by_key = {}
    for i, r in enumerate(m.itertuples(index=False)):
        idx_by_key.setdefault((r.split, r.image_file), []).append((i, r))
    for (sp, imf), rows in idx_by_key.items():
        if not isinstance(imf, str):
            continue
        img = cv2.imread(os.path.join(DATASET_DIR, sp, "images", imf), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for i, r in rows:
            crop, _ = extract_crop(img, r.xc, r.yc, r.w, r.h, pad=0.25, img_size=IMG_SIZE)
            if crop.size == 0 or min(crop.shape[:2]) < 6:
                continue
            gst_ax[i] = gst_orientation(crop)[0]
            hand[i] = hand_features(crop, max(r.w, r.h) * IMG_SIZE)
    m["gst_axis"] = gst_ax
    m["axis_err_vs_tracking"] = circ_dist(m.human_axis, m.trk_axis, 180)
    m["gst_err_vs_human"] = circ_dist(m.gst_axis, m.human_axis, 180)

    section("2. LINCHPIN: human axis vs tracking axis (trusted movers, human gave a line)")
    val = m[(m.trusted == True) & m.has_axis & m.axis_err_vs_tracking.notna()]  # noqa: E712

    def rep(d, name):
        if len(d) == 0:
            print(f"  {name}: n=0"); return
        e = d.axis_err_vs_tracking
        print(f"  {name}: n={len(d):>4}  median={e.median():4.1f}  mean={e.mean():4.1f}"
              f"  Acc@45={np.mean(e <= 45):.2f}  Acc@30={np.mean(e <= 30):.2f}")
    rep(val, "all trusted-mover crops with a line")
    rep(val[val.moving_human], "  + human says moving")
    rep(val[val.moving_human & (val.scc <= 1)], "  + single-animal frame")
    rng = np.random.default_rng(0)
    rnd = circ_dist(rng.uniform(0, 180, len(val)), val.trk_axis, 180)
    print(f"  [random baseline: median={np.median(rnd):.1f}  Acc@45={np.mean(rnd <= 45):.2f}]")

    section("3. SINGLE-IMAGE GST vs HUMAN axis (all crops where the human gave a line)")
    gv = m[m.has_axis & m.gst_err_vs_human.notna()]
    e = gv.gst_err_vs_human
    print(f"  n={len(gv)}  median={e.median():.1f}  Acc@45={np.mean(e <= 45):.2f}")
    gh = gv[gv.has_head]
    print(f"  (where the human was confident enough to give a head, n={len(gh)}: "
          f"median={gh.gst_err_vs_human.median():.1f})")

    section("4. HEAD-DISCERNIBILITY RATE")
    mv = m[m.moving_human]
    print(f"  overall: head={L.has_head.mean()*100:.0f}%  axis-only={((dc>=10)&(dc<20)).mean()*100:.0f}%"
          f"  none={(dc==-2).mean()*100:.0f}%")
    print(f"  among human-moving: head={mv.has_head.mean()*100:.0f}%"
          f"  axis-only={((mv.direction_class>=10)&(mv.direction_class<20)).mean()*100:.0f}%"
          f"  none={(mv.direction_class==-2).mean()*100:.0f}%")

    section("5. MOVING vs STATIONARY on REAL labels (flight-disjoint)")
    d = m[m.motion_state.isin(("stationary",) + MOVING) & m.image_file.notna()].copy()
    keep = [i for i in d.index if i in hand]
    d = d.loc[keep]
    X = np.vstack([hand[i] for i in d.index])
    y = d.moving_human.astype(int).to_numpy()
    groups = d.flight_id.to_numpy()
    print(f"  crops={len(d)}  moving={int(y.sum())}  stationary={int((y == 0).sum())}  flights={d.flight_id.nunique()}")

    def cv(make):
        a = []
        for s in (0, 1, 2):
            for tr, te in GroupKFold(5).split(X, y, groups):
                if len(np.unique(y[tr])) < 2:
                    continue
                mdl = make(s); mdl.fit(X[tr], y[tr])
                a.append(balanced_accuracy_score(y[te], mdl.predict(X[te])))
        return np.mean(a), np.std(a)
    rf = cv(lambda s: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=s))
    lr = cv(lambda s: make_pipeline(StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=s)))
    maj = d.groupby("flight_id").moving_human.transform(lambda s: int(round(s.mean())))
    print(f"  LogReg bal-acc={lr[0]:.2f}+-{lr[1]:.2f}  RandomForest bal-acc={rf[0]:.2f}+-{rf[1]:.2f}"
          f"  majority=0.50  scene-ceiling={balanced_accuracy_score(y, maj):.2f}")

    out = OUTPUT_DIR / "label_validation.csv"
    cols = ["crop_id", "motion_state", "moving_human", "direction_class", "has_axis", "has_head",
            "human_axis", "human_full", "scc", "tracklet_id", "trusted", "trk_axis", "trk_full",
            "axis_err_vs_tracking", "gst_axis", "gst_err_vs_human"]
    m[cols].to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
