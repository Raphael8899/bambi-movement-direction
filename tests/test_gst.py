"""Tests for the gradient structure tensor orientation estimator.

Returns the dominant STRUCTURE long-axis in image coords:
0 deg = horizontal (along +x / columns), 90 deg = vertical (along rows),
range [0, 180) (axial). Coherence in [0,1] = anisotropy/confidence.
"""
import numpy as np
import pytest

from src.gst import gst_orientation
from src.metrics_circular import circ_dist


def _elong_blob(size=80, sx=14.0, sy=3.5, angle_deg=0.0):
    """A Gaussian blob elongated along its long axis, rotated by angle_deg (array coords)."""
    yy, xx = np.mgrid[-size // 2:size // 2, -size // 2:size // 2].astype(float)
    th = np.deg2rad(angle_deg)
    xr = xx * np.cos(th) + yy * np.sin(th)
    yr = -xx * np.sin(th) + yy * np.cos(th)
    g = np.exp(-(xr ** 2 / (2 * sx ** 2) + yr ** 2 / (2 * sy ** 2)))
    return (g * 255).astype(np.uint8)


def test_horizontal_blob_orientation_zero():
    ori, coh = gst_orientation(_elong_blob(angle_deg=0))
    assert circ_dist(ori, 0.0, period=180) < 7.0
    assert coh > 0.5

def test_vertical_blob_orientation_ninety():
    ori, coh = gst_orientation(_elong_blob(angle_deg=90))
    assert circ_dist(ori, 90.0, period=180) < 7.0

def test_diagonal_blob_orientation():
    ori, _ = gst_orientation(_elong_blob(angle_deg=30))
    assert circ_dist(ori, 30.0, period=180) < 8.0

def test_isotropic_blob_low_coherence():
    _, coh = gst_orientation(_elong_blob(sx=8.0, sy=8.0))  # round blob
    assert coh < 0.2

def test_orientation_in_axial_range():
    ori, _ = gst_orientation(_elong_blob(angle_deg=120))
    assert 0.0 <= ori < 180.0
