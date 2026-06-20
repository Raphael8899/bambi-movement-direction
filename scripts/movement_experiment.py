"""Moving-vs-stationary bake-off across feature families, evaluated flight-disjoint.

Builds a labelled crop set from the tracking proxy labels (see src.movement), then scores
five feature families with the same flight-grouped CV so no flight is in both train and
test:

  classical hand features  -> logistic regression and random forest
  small CNN from scratch    -> trained per fold
  frozen DINOv2 / CLIP / BioCLIP -> logistic head on cached features

Reports accuracy, balanced accuracy and ROC-AUC (mean +/- std over folds x seeds) against
the majority-class baseline. Writes output/movement_results.csv and prints a table.

The labels are weak: MOVING tracklets are the ones the tracker trusted, STATIONARY are
untrusted tracklets that barely moved. That is proxy supervision, partly circular with
the tracker -- treat the numbers as a first pass, not gold.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DATASET_DIR, IMG_SIZE, OUTPUT_DIR
from src.data_loader import load_annotations
from src.deep_features import cached_features, train_cnn
from src.movement import (
    FEATURE_NAMES, crop_from_row, hand_features, proxy_labels, sample_member_crops,
)
from src.tracking import build_tracklets

OUT_CSV = OUTPUT_DIR / "movement_results.csv"
FEAT_CACHE = OUTPUT_DIR / "deep_feat_cache"
SEEDS = [0, 1, 2]
N_FOLDS = 5


def build_crop_set():
    """Sampled detections with proxy labels; one row per crop to extract."""
    track = pd.read_csv(OUTPUT_DIR / "tracking_directions.csv")
    labels = proxy_labels(track)

    df = load_annotations()
    df = df[df.flight_id != -1].copy()
    tracked = build_tracklets(df, img_size=IMG_SIZE)
    samples = sample_member_crops(tracked, labels)
    return samples


def load_crops(samples):
    """Extract every crop, grouping by image so each frame is decoded once.

    Returns (crops_gray list, kept samples DataFrame, hand feature matrix). Drops crops
    that come out empty.
    """
    crops, feats, keep = [], [], []
    for image_file, g in samples.groupby("image_file"):
        split = g.iloc[0]["split"]
        path = os.path.join(DATASET_DIR, split, "images", image_file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for r in g.itertuples(index=False):
            crop = crop_from_row(img, r, img_size=IMG_SIZE)
            if crop.size == 0 or min(crop.shape[:2]) < 6:
                continue
            long_px = max(r.w, r.h) * IMG_SIZE
            crops.append(crop)
            feats.append(hand_features(crop, long_px))
            keep.append(r._asdict())

    kept = pd.DataFrame(keep)
    return crops, kept, np.vstack(feats)


def crop_ids(kept):
    """Stable per-crop id for the feature cache (image + box center)."""
    return [f"{r.image_file}:{r.xc:.5f}:{r.yc:.5f}" for r in kept.itertuples(index=False)]


def _scores(y_true, prob, pred):
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan")
    return (accuracy_score(y_true, pred),
            balanced_accuracy_score(y_true, pred),
            auc)


def eval_sklearn(make_model, X, y, groups, seeds=SEEDS, n_folds=N_FOLDS):
    """Flight-grouped CV over several seeds for an sklearn classifier factory."""
    rows = []
    for seed in seeds:
        gkf = GroupKFold(n_splits=n_folds)
        for tr, te in gkf.split(X, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            model = make_model(seed)
            model.fit(X[tr], y[tr])
            prob = model.predict_proba(X[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            rows.append(_scores(y[te], prob, pred))
    return np.array(rows)


def eval_cnn(crops, y, groups, seeds=SEEDS, n_folds=N_FOLDS):
    """Flight-grouped CV for the from-scratch CNN, trained per fold."""
    crops = np.array(crops, dtype=object)
    rows = []
    for seed in seeds:
        gkf = GroupKFold(n_splits=n_folds)
        for tr, te in gkf.split(crops, y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            prob = train_cnn(list(crops[tr]), y[tr], list(crops[te]), y[te], seed=seed)
            pred = (prob >= 0.5).astype(int)
            rows.append(_scores(y[te], prob, pred))
    return np.array(rows)


def majority_baseline(y, groups, n_folds=N_FOLDS):
    """Predict the train-fold majority class on every test fold."""
    rows = []
    gkf = GroupKFold(n_splits=n_folds)
    for tr, te in gkf.split(y, y, groups):
        maj = int(np.round(y[tr].mean()))
        pred = np.full(len(te), maj)
        acc = accuracy_score(y[te], pred)
        bacc = balanced_accuracy_score(y[te], pred)
        rows.append((acc, bacc, float("nan")))
    return np.array(rows)


def summarize(name, arr):
    # nan-aware per column; a fully-nan column (AUC of the majority baseline) stays nan
    def col(fn, j):
        c = arr[:, j][~np.isnan(arr[:, j])]
        return float(fn(c)) if c.size else float("nan")
    m = [col(np.mean, j) for j in range(arr.shape[1])]
    s = [col(np.std, j) for j in range(arr.shape[1])]
    return {
        "family": name,
        "accuracy": round(m[0], 3), "accuracy_std": round(s[0], 3),
        "balanced_acc": round(m[1], 3), "balanced_acc_std": round(s[1], 3),
        "roc_auc": round(m[2], 3), "roc_auc_std": round(s[2], 3),
    }


def main():
    t0 = time.time()
    samples = build_crop_set()
    n_mov_t = samples[samples.label == 1].tracklet_id.nunique()
    n_sta_t = samples[samples.label == 0].tracklet_id.nunique()
    print(f"labelled tracklets: {n_mov_t} moving, {n_sta_t} stationary")

    crops, kept, Xhand = load_crops(samples)
    y = kept["label"].to_numpy().astype(int)
    groups = kept["flight_id"].to_numpy()
    print(f"crops: {len(crops)}  moving={int((y == 1).sum())}  stationary={int((y == 0).sum())}  "
          f"flights={len(np.unique(groups))}  ({time.time() - t0:.0f}s)")

    ids = crop_ids(kept)
    results = []

    results.append(summarize("majority_baseline", majority_baseline(y, groups)))

    results.append(summarize("hand_logreg", eval_sklearn(
        lambda s: make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced",
                                                   random_state=s)),
        Xhand, y, groups)))
    results.append(summarize("hand_randomforest", eval_sklearn(
        lambda s: RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         random_state=s),
        Xhand, y, groups)))

    print(f"running CNN ({time.time() - t0:.0f}s)...", flush=True)
    results.append(summarize("cnn_scratch", eval_cnn(crops, y, groups)))

    for kind in ("dinov2", "clip", "bioclip"):
        print(f"extracting {kind} features ({time.time() - t0:.0f}s)...", flush=True)
        feats = cached_features(kind, crops, ids, FEAT_CACHE)
        results.append(summarize(f"frozen_{kind}", eval_sklearn(
            lambda s: make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=2000,
                                                       class_weight="balanced",
                                                       random_state=s)),
            feats, y, groups)))

    res = pd.DataFrame(results)
    res.to_csv(OUT_CSV, index=False)

    cols = ["family", "accuracy", "accuracy_std", "balanced_acc", "balanced_acc_std",
            "roc_auc", "roc_auc_std"]
    print(f"\nwrote {OUT_CSV}\n")
    print(res[cols].to_string(index=False))
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
