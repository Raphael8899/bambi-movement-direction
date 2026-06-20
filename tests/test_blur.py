"""Tests for single-image motion-blur axis estimators.

Each estimator returns (axis_deg in [0,180), confidence). The axis is the direction of
motion in image coords: 0 = horizontal (+x), 90 = vertical. We synthesise blur by
convolving a patch with a linear motion kernel at a known angle and check recovery.
"""
import numpy as np
import pytest

from src.blur import (
    blur_axis_gradient, blur_axis_spectrum, blur_axis_cepstrum, motion_kernel,
)
from src.metrics_circular import circ_dist


def _motion_blur(img, length, angle_deg):
    import cv2
    k = motion_kernel(length, angle_deg)
    return cv2.filter2D(img.astype(np.float32), -1, k)


def _blob(size=128, sx=6.0, sy=6.0):
    yy, xx = np.mgrid[-size // 2:size // 2, -size // 2:size // 2].astype(float)
    g = np.exp(-(xx ** 2 / (2 * sx ** 2) + yy ** 2 / (2 * sy ** 2)))
    return (g * 255).astype(np.float32)


def _textured_patch(size=128, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(128, 40, (size, size))
    import cv2
    return np.clip(cv2.GaussianBlur(base, (0, 0), 1.5), 0, 255).astype(np.float32)


# A long, clear blur over a textured patch is the easy case for gradient/spectrum.
@pytest.mark.parametrize("angle", [0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
def test_gradient_recovers_angle(angle):
    img = _motion_blur(_textured_patch(seed=1), length=21, angle_deg=angle)
    axis, conf = blur_axis_gradient(img)
    assert 0.0 <= axis < 180.0
    assert circ_dist(axis, angle, period=180) < 15.0
    assert conf > 0.0


@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 135.0])
def test_spectrum_recovers_angle(angle):
    img = _motion_blur(_textured_patch(seed=2), length=25, angle_deg=angle)
    axis, conf = blur_axis_spectrum(img)
    assert 0.0 <= axis < 180.0
    assert circ_dist(axis, angle, period=180) < 15.0


@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 135.0])
def test_cepstrum_recovers_angle(angle):
    img = _motion_blur(_textured_patch(seed=3), length=25, angle_deg=angle)
    axis, conf, length = blur_axis_cepstrum(img)
    assert 0.0 <= axis < 180.0
    # cepstrum is the noisiest; allow a wide tolerance
    assert circ_dist(axis, angle, period=180) < 30.0


def test_gradient_confidence_low_on_isotropic():
    iso = _blob(sx=8.0, sy=8.0)  # round, no motion
    blurred = _motion_blur(_textured_patch(seed=4), length=21, angle_deg=30.0)
    _, c_iso = blur_axis_gradient(iso)
    _, c_blur = blur_axis_gradient(blurred)
    assert c_iso < c_blur


def test_spectrum_confidence_low_on_isotropic():
    iso = _blob(sx=8.0, sy=8.0)
    blurred = _motion_blur(_textured_patch(seed=5), length=25, angle_deg=60.0)
    _, c_iso = blur_axis_spectrum(iso)
    _, c_blur = blur_axis_spectrum(blurred)
    assert c_iso < c_blur


def test_outputs_in_axial_range():
    img = _motion_blur(_textured_patch(seed=6), length=19, angle_deg=100.0)
    for fn in (blur_axis_gradient, blur_axis_spectrum):
        axis, _ = fn(img)
        assert 0.0 <= axis < 180.0
    axis, _, _ = blur_axis_cepstrum(img)
    assert 0.0 <= axis < 180.0


def test_motion_kernel_is_normalised():
    k = motion_kernel(15, 30.0)
    assert abs(k.sum() - 1.0) < 1e-6
    assert k.shape[0] == k.shape[1]


def test_cepstrum_length_tracks_blur():
    short = _motion_blur(_textured_patch(seed=7), length=11, angle_deg=0.0)
    long = _motion_blur(_textured_patch(seed=7), length=31, angle_deg=0.0)
    _, _, len_short = blur_axis_cepstrum(short)
    _, _, len_long = blur_axis_cepstrum(long)
    assert len_long > len_short
