"""Tests for the proxy-label rule and the classical moving-vs-stationary features.

The proxy label is derived from the tracker: MOVING = trusted heading, STATIONARY =
untrusted but barely displaced over enough frames. The hand features should rank a sharp
elongated blob (stationary body) apart from a motion-smeared one (moving).
"""
import numpy as np
import pandas as pd
import pytest

from src.deep_features import prep_cnn_batch, to_three_channel
from src.movement import (
    FEATURE_NAMES, hand_features, inner_surround_sharpness, is_moving_label,
    proxy_labels, sample_member_crops,
)


def _blob(size=80, sx=14.0, sy=3.5, angle_deg=0.0):
    yy, xx = np.mgrid[-size // 2:size // 2, -size // 2:size // 2].astype(float)
    th = np.deg2rad(angle_deg)
    xr = xx * np.cos(th) + yy * np.sin(th)
    yr = -xx * np.sin(th) + yy * np.cos(th)
    g = np.exp(-(xr ** 2 / (2 * sx ** 2) + yr ** 2 / (2 * sy ** 2)))
    return (g * 255).astype(np.uint8)


def _motion_blur(img, length, angle_deg=0.0):
    import cv2
    from src.blur import motion_kernel
    k = motion_kernel(length, angle_deg)
    return np.clip(cv2.filter2D(img.astype(np.float32), -1, k), 0, 255).astype(np.uint8)


def test_trusted_is_moving():
    assert is_moving_label(True, disp_px=500.0, n_obs=60) == 1


def test_untrusted_small_disp_is_stationary():
    assert is_moving_label(False, disp_px=8.0, n_obs=10) == 0


def test_untrusted_but_displaced_is_unlabelled():
    # moved a lot but the tracker didn't trust it -> ambiguous, drop it
    assert is_moving_label(False, disp_px=200.0, n_obs=10) is None


def test_too_few_obs_is_unlabelled():
    assert is_moving_label(False, disp_px=3.0, n_obs=2) is None


def test_proxy_labels_keeps_only_labelled():
    track = pd.DataFrame({
        "tracklet_id": [0, 1, 2, 3],
        "trusted": [True, False, False, False],
        "disp_px": [300.0, 5.0, 400.0, 2.0],
        "n_obs": [40, 9, 9, 3],          # 0 moving, 1 stationary, 2 displaced, 3 short
    })
    out = proxy_labels(track)
    assert dict(zip(out.tracklet_id, out.label)) == {0: 1, 1: 0}


def test_sample_member_crops_caps_per_tracklet():
    det = pd.DataFrame({
        "tracklet_id": [5] * 6,
        "flight_id": [3] * 6,
        "frame_num": list(range(6)),
        "split": ["train"] * 6,
        "image_file": [f"3_{i}_jpg.jpg" for i in range(6)],
        "xc": [0.5] * 6, "yc": [0.5] * 6, "w": [0.05] * 6, "h": [0.05] * 6,
    })
    labels = pd.DataFrame({"tracklet_id": [5], "label": [1]})
    out = sample_member_crops(det, labels, max_per_tracklet=4)
    assert len(out) == 4
    # mid-frames, not the entering/leaving ends
    assert set(out.frame_num) == {1, 2, 3, 4}


def test_sample_member_crops_skips_unlabelled_tracklets():
    det = pd.DataFrame({
        "tracklet_id": [5, 9],
        "flight_id": [3, 3],
        "frame_num": [0, 0],
        "split": ["train", "train"],
        "image_file": ["a.jpg", "b.jpg"],
        "xc": [0.5, 0.5], "yc": [0.5, 0.5], "w": [0.05, 0.05], "h": [0.05, 0.05],
    })
    labels = pd.DataFrame({"tracklet_id": [5], "label": [0]})
    out = sample_member_crops(det, labels)
    assert out.tracklet_id.tolist() == [5]


def test_hand_features_length_and_order():
    f = hand_features(_blob(), box_long_px=40.0)
    assert f.shape == (len(FEATURE_NAMES),)
    assert f[-1] == pytest.approx(40.0)  # size feature passed through


def test_inner_sharper_than_surround_for_clean_blob():
    # a crisp blob on flat background has more Laplacian energy inside than out
    crop = _blob(sx=10.0, sy=10.0)
    assert inner_surround_sharpness(crop) > 1.0


def test_motion_blur_lowers_inner_sharpness():
    sharp = _blob(sx=10.0, sy=4.0)
    smeared = _motion_blur(sharp, length=15, angle_deg=0.0)
    assert inner_surround_sharpness(smeared) < inner_surround_sharpness(sharp)


def _textured_patch(size=80, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 40, (size, size))
    return np.clip(cv2.GaussianBlur(base, (0, 0), 1.5), 0, 255).astype(np.uint8)


def test_blur_length_is_nonnegative():
    # the cepstral length is the noisiest cue; here we only pin its sign and finiteness
    i_blur = FEATURE_NAMES.index("blur_length")
    for c in (_textured_patch(seed=7), _motion_blur(_textured_patch(seed=7), 25)):
        v = hand_features(c, 40.0)[i_blur]
        assert np.isfinite(v) and v >= 0.0


def test_hand_features_finite():
    for c in (_blob(), _motion_blur(_blob(), 15), np.zeros((40, 40), np.uint8)):
        assert np.all(np.isfinite(hand_features(c, 30.0)))


def test_to_three_channel_shape_and_range():
    out = to_three_channel(_blob(size=50), size=224)
    assert out.shape == (3, 224, 224)
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert np.allclose(out[0], out[1]) and np.allclose(out[1], out[2])  # replicated gray


def test_prep_cnn_batch_shape():
    crops = [_blob(size=40), _blob(size=60)]
    t = prep_cnn_batch(crops, size=64)
    assert tuple(t.shape) == (2, 1, 64, 64)
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0
