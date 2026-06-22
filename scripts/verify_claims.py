"""Re-derive every headline number from the committed result CSVs and check it against the
values stated in the docs. Needs only pandas + numpy (no raw dataset, no OpenCV) -- so it runs
anywhere the repo is cloned, and answers one question: do the documented claims match the data?

    python scripts/verify_claims.py

Exits non-zero if any check fails.
"""
import os
import sys

import numpy as np
import pandas as pd

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
_fail = [0]


def check(name, got, exp, tol=0.0):
    ok = (got == exp) if tol == 0 else (got is not None and abs(float(got) - float(exp)) <= tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<46} got={got}  exp={exp}")
    if not ok:
        _fail[0] += 1


# --- circular helpers (axial, period=180), inlined to avoid heavy deps ---
def circ_dist(a, b, p=180.0):
    d = np.abs((np.asarray(a, float) - np.asarray(b, float)) % p)
    return np.minimum(d, p - d)


def circ_mean(ang, p=180.0):
    ph = np.asarray(ang, float) * (2 * np.pi / p)
    m = np.arctan2(np.sin(ph).mean(), np.cos(ph).mean())   # mean phase in radians
    return (m * (p / (2 * np.pi))) % p                     # back to input units, in [0, p)


def is_trusted(r):
    return (r.n_steps >= 5 and r.median_inlier >= 0.5 and r.R >= 0.5
            and r.rayleigh_p < 0.05 and r.disp_px >= 5.0)


print("\n=== DATASET / EDA (from output/eda_stats.csv) ===")
eda = pd.read_csv(f"{OUT}/eda_stats.csv").sort_values("cls")
check("per-class box counts", eda.n_boxes.tolist(), [21787, 17403, 6856])
check("shrink ratio class 0/1/2 (0.81/1.15/0.37)",
      [round(v, 2) for v in eda.shrink_ratio_median], [0.81, 1.15, 0.37])
check("median intensity class 2 (boar, coolest)", round(eda.mean_int_median.iloc[2], 0), 79, tol=1)

print("\n=== TRACKING (from output/tracking_directions.csv) ===")
td = pd.read_csv(f"{OUT}/tracking_directions.csv")
check("total tracklets", len(td), 2697)
recomp = td.apply(is_trusted, axis=1)
check("recomputed-vs-stored trusted mismatches", int((recomp != td.trusted).sum()), 0)
tr = td[td.trusted]
check("trusted total", len(tr), 190)
check("trusted by class", tr.cls.value_counts().sort_index().tolist(), [83, 68, 39])
core = tr[(tr.n_steps >= 8) & (tr.disp_px >= 50)]
check("high-confidence core (>=8 steps & >=50px)", len(core), 138)
check("core by class", core.cls.value_counts().sort_index().tolist(), [58, 49, 31])
check("trusted with >1000px displacement", int((tr.disp_px > 1000).sum()), 18)
check("all-tracklet median inlier (~0.86)", round(td.median_inlier.median(), 2), 0.86, tol=0.01)

print("\n=== SINGLE-IMAGE AXIS (from output/blur_eval.csv) ===")
be = pd.read_csv(f"{OUT}/blur_eval.csv")
mov = be[be.group == "mover"]
check("mover crops", len(mov), 945)
check("distinct mover tracklets (independent dirs)", mov.tracklet_id.nunique(), 189)
g = mov.gst_err.dropna()
check("GST median axial error (~29.1)", round(g.median(), 1), 29.1, tol=0.2)
check("GST Acc@45 (~0.68)", round((g <= 45).mean(), 2), 0.68, tol=0.02)
pcb = []
for c, exp_g, exp_c in [(0, 39.4, 16.1), (1, 25.4, 32.6), (2, 19.3, 36.2)]:
    cd = mov[mov.cls == c]
    d_c = circ_dist(np.full(len(cd), circ_mean(cd.gt_axis)), cd.gt_axis)
    pcb.append(d_c)
    check(f"class {c}: GST median", round(cd.gst_err.median(), 1), exp_g, tol=0.3)
    check(f"class {c}: constant-baseline median", round(float(np.median(d_c)), 1), exp_c, tol=0.3)
# pooled per-class baseline = the consistent overall bar (beats GST 29.1)
check("overall per-class-mean baseline (~25.0)", round(float(np.median(np.concatenate(pcb))), 1), 25.0, tol=0.3)

print("\n=== MOVING/STATIONARY (from output/movement_results.csv) ===")
mr = pd.read_csv(f"{OUT}/movement_results.csv").set_index("family")
for fam, exp in [("frozen_bioclip", 0.64), ("frozen_dinov2", 0.63), ("frozen_clip", 0.62),
                 ("hand_randomforest", 0.58), ("cnn_scratch", 0.50), ("majority_baseline", 0.50)]:
    check(f"{fam} balanced acc", round(mr.loc[fam, "balanced_acc"], 2), exp, tol=0.015)

lv_path = f"{OUT}/label_validation.csv"
if os.path.exists(lv_path):
    print("\n=== HUMAN-LABEL VALIDATION (from output/label_validation.csv) ===")
    lv = pd.read_csv(lv_path)
    for c in ("has_axis", "has_head", "trusted", "moving_human"):
        lv[c] = lv[c].astype(str).isin(("True", "1", "1.0"))
    link = lv[lv.trusted & lv.has_axis & lv.axis_err_vs_tracking.notna()]
    check("linchpin: human-vs-tracking axis median (~22.7)", round(link.axis_err_vs_tracking.median(), 1), 22.7, tol=0.6)
    gv = lv[lv.has_axis & lv.gst_err_vs_human.notna()]
    check("GST vs human axis median (~10.7)", round(gv.gst_err_vs_human.median(), 1), 10.7, tol=0.8)
    check("head-discernibility rate (~0.14)", round(lv.has_head.mean(), 2), 0.14, tol=0.02)

ls_path = f"{OUT}/label_validation_summary.csv"
if os.path.exists(ls_path):
    print("\n=== VALIDATION SUMMARY (from output/label_validation_summary.csv) ===")
    s = pd.read_csv(ls_path).iloc[0]
    check("real-label moving/stationary LogReg (~0.62)", round(s.real_logreg_balacc, 2), 0.62, tol=0.03)
    check("real-label scene-ceiling (~0.78)", round(s.real_scene_ceiling, 2), 0.78, tol=0.03)
    check("motion binary AUC (~0.66)", round(s.motion_binary_auc, 2), 0.66, tol=0.03)
    check("motion slight-vs-stronger = chance (~0.52)", round(s.motion_slight_vs_stronger, 2), 0.52, tol=0.04)
    check("motion 3-level (~0.43)", round(s.motion_3level, 2), 0.43, tol=0.04)

print("\n" + ("=" * 60))
if _fail[0] == 0:
    print("ALL CHECKS PASSED — the documented numbers match the data.")
else:
    print(f"{_fail[0]} CHECK(S) FAILED — investigate before trusting the docs.")
sys.exit(1 if _fail[0] else 0)
