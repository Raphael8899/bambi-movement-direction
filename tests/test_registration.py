"""Tests for background registration on a synthetic shifted+rotated image pair."""
import cv2
import numpy as np

from src.registration import estimate_transform, apply_transform


def _textured_image(size=512, seed=0):
    # ORB needs corners/texture; random blobs on a low-contrast background mimic the
    # thermal frames well enough to register.
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 120, np.uint8)
    for _ in range(120):
        x, y = rng.integers(20, size - 20, 2)
        r = int(rng.integers(4, 14))
        val = int(rng.integers(150, 255))
        cv2.circle(img, (int(x), int(y)), r, val, -1)
    return img


def _warp_full(img, angle_deg, tx, ty):
    h, w = img.shape
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    R[0, 2] += tx
    R[1, 2] += ty
    return cv2.warpAffine(img, R, (w, h), borderMode=cv2.BORDER_REFLECT)


def test_recovers_known_shift_and_rotation():
    img_a = _textured_image()
    # known ego-motion: rotate 3 deg about the centre and shift (+40, -25) px
    img_b = _warp_full(img_a, angle_deg=3.0, tx=40.0, ty=-25.0)

    M, inlier_ratio = estimate_transform(img_a, img_b)
    assert M is not None
    assert inlier_ratio > 0.5

    # a point mapped A->B then compared to where the same warp puts it
    h, w = img_a.shape
    R = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
    R[0, 2] += 40.0
    R[1, 2] += -25.0
    for px, py in [(100.0, 100.0), (400.0, 300.0), (256.0, 256.0)]:
        ex = R[0, 0] * px + R[0, 1] * py + R[0, 2]
        ey = R[1, 0] * px + R[1, 1] * py + R[1, 2]
        gx, gy = apply_transform(M, px, py)
        assert abs(gx - ex) < 3.0
        assert abs(gy - ey) < 3.0


def test_pure_translation_recovered():
    img_a = _textured_image(seed=1)
    img_b = _warp_full(img_a, angle_deg=0.0, tx=30.0, ty=15.0)
    M, _ = estimate_transform(img_a, img_b)
    assert M is not None
    gx, gy = apply_transform(M, 200.0, 200.0)
    assert abs(gx - 230.0) < 3.0
    assert abs(gy - 215.0) < 3.0


def test_too_few_features_returns_none():
    flat = np.full((256, 256), 100, np.uint8)
    M, ratio = estimate_transform(flat, flat.copy())
    assert M is None
    assert ratio == 0.0
