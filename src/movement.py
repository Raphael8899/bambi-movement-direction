"""Single-crop moving-vs-stationary features and the tracking-derived proxy labels.

The question is whether one AOS crop carries enough signal to tell a moving animal
(motion-smeared) from a stationary one (sharp body). We have no movement labels yet, so
we borrow weak labels from the tracking output: tracklets the direction pipeline TRUSTED
are moving, tracklets it left untrusted that also barely displaced are stationary. This
is proxy supervision and partly circular with the tracker -- to be re-checked against
human labels later.

Hand features (one vector per crop): gst coherence, blob eccentricity, inner-vs-surround
sharpness, cepstral blur length, contrast-to-surround, size. Axis convention follows
src.gst / src.bb_refinement. Assumes the animal is the bright (warm) blob.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.bb_refinement import refine_box, segment_warm_blob
from src.blur import blur_axis_cepstrum
from src.crops import contrast_to_surround, extract_crop
from src.data_loader import yolo_to_box
from src.gst import gst_orientation

# Proxy-label thresholds; mirror the tracking trust rule for the stationary side.
STATIONARY_MAX_DISP_PX = 15.0
STATIONARY_MIN_OBS = 5
MAX_CROPS_PER_TRACKLET = 4

FEATURE_NAMES = [
    "coherence", "eccentricity", "inner_surround_sharp",
    "blur_length", "contrast_surround", "size_px",
]


def is_moving_label(trusted: bool, disp_px: float, n_obs: int,
                    max_disp: float = STATIONARY_MAX_DISP_PX,
                    min_obs: int = STATIONARY_MIN_OBS):
    """Proxy label for one tracklet: 1 moving, 0 stationary, None if neither.

    Moving = the tracker trusted the heading. Stationary = untrusted but with a small net
    displacement over enough observations. Everything else (untrusted but displaced, too
    short) is unlabelled.
    """
    if trusted:
        return 1
    if (not trusted) and disp_px < max_disp and n_obs >= min_obs:
        return 0
    return None


def proxy_labels(track_df):
    """Per-tracklet proxy labels from a tracking_directions-style frame.

    Needs columns tracklet_id, trusted, disp_px, n_obs. Returns a DataFrame with
    tracklet_id and label (0/1) for the labelled tracklets only.
    """
    import pandas as pd

    rows = []
    for r in track_df.itertuples(index=False):
        lab = is_moving_label(bool(r.trusted), float(r.disp_px), int(r.n_obs))
        if lab is not None:
            rows.append({"tracklet_id": int(r.tracklet_id), "label": lab})
    return pd.DataFrame(rows, columns=["tracklet_id", "label"])


def sample_member_crops(detections, labels, max_per_tracklet=MAX_CROPS_PER_TRACKLET):
    """Pick up to N mid-frame detections per labelled tracklet.

    detections: the build_tracklets output (one row per detection, has tracklet_id,
    flight_id, frame_num...). labels: proxy_labels output. Returns one row per chosen
    detection with its tracklet label and flight, mid-frames preferred over the ends
    (entering/leaving the frame is the noisiest part of a track).
    """
    import pandas as pd

    lab = labels.set_index("tracklet_id")["label"]
    chosen = []
    for tid, g in detections.groupby("tracklet_id"):
        if tid not in lab.index:
            continue
        g = g.sort_values("frame_num")
        n = len(g)
        if n <= max_per_tracklet:
            picks = g
        else:
            lo = (n - max_per_tracklet) // 2
            picks = g.iloc[lo:lo + max_per_tracklet]
        for r in picks.itertuples(index=False):
            chosen.append({
                "tracklet_id": int(tid),
                "flight_id": int(r.flight_id),
                "label": int(lab.loc[tid]),
                "split": r.split,
                "image_file": r.image_file,
                "frame_num": int(r.frame_num),
                "xc": r.xc, "yc": r.yc, "w": r.w, "h": r.h,
            })
    return pd.DataFrame(chosen)


def inner_surround_sharpness(crop_gray, mask=None):
    """Laplacian-variance ratio of the warm blob vs the ring around it.

    A sharp, in-focus animal has more high-frequency energy inside the blob than its
    background; motion blur flattens the inside, dropping the ratio. Returns inner var
    divided by surround var (1.0 if either region is empty).
    """
    g = crop_gray.astype(np.float32)
    if mask is None:
        mask = segment_warm_blob(crop_gray.astype(np.uint8))
    if mask is None:
        return 1.0
    m = mask.astype(bool)
    if m.sum() < 4 or (~m).sum() < 4:
        return 1.0
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    inner = lap[m].var()
    surround = lap[~m].var()
    if surround <= 1e-6:
        return 1.0
    return float(inner / surround)


def hand_features(crop_gray, box_long_px):
    """The six classical features for one crop, in FEATURE_NAMES order.

    crop_gray: square grayscale crop around the animal. box_long_px: long side of the
    original (image-scale) box, used as the size feature.
    """
    crop_gray = np.ascontiguousarray(crop_gray)
    coherence = gst_orientation(crop_gray)[1]

    rb = refine_box(crop_gray)
    ecc = rb.eccentricity if rb.success else 0.0
    mask = None
    if rb.success:
        mask = np.zeros(crop_gray.shape[:2], np.uint8)
        mask[rb.y0:rb.y1, rb.x0:rb.x1] = 1  # blob bbox is enough for the inner/outer split

    sharp = inner_surround_sharpness(crop_gray, mask)
    blur_len = blur_axis_cepstrum(crop_gray)[2]

    # contrast-to-surround on the crop itself: blob mean minus a border ring.
    h, w = crop_gray.shape[:2]
    bw = max(2, min(h, w) // 8)
    border = np.concatenate([
        crop_gray[:bw, :].ravel(), crop_gray[-bw:, :].ravel(),
        crop_gray[:, :bw].ravel(), crop_gray[:, -bw:].ravel(),
    ])
    contrast = float(crop_gray.astype(np.float32).mean() - border.mean())

    return np.array([
        coherence, ecc, sharp, blur_len, contrast, float(box_long_px),
    ], dtype=np.float32)


def crop_from_row(image, row, pad=0.25, img_size=2048):
    """Square grayscale crop for one sampled detection row (from sample_member_crops)."""
    crop, _ = extract_crop(image, row.xc, row.yc, row.w, row.h, pad=pad, img_size=img_size)
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop
