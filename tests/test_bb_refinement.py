"""Tests for warm-blob segmentation + box refinement."""
import numpy as np
import pytest

from src.bb_refinement import refine_box
from src.metrics_circular import circ_dist


def _crop_with_blob(W=120, H=120, sx=15.0, sy=4.0, angle_deg=0.0, bg=20, amp=200):
    """Dark crop with one bright elongated Gaussian blob in the centre."""
    yy, xx = np.mgrid[-H // 2:H // 2, -W // 2:W // 2].astype(float)
    th = np.deg2rad(angle_deg)
    xr = xx * np.cos(th) + yy * np.sin(th)
    yr = -xx * np.sin(th) + yy * np.cos(th)
    g = np.exp(-(xr ** 2 / (2 * sx ** 2) + yr ** 2 / (2 * sy ** 2)))
    img = bg + amp * g
    return np.clip(img, 0, 255).astype(np.uint8)


def test_refine_tightens_box():
    r = refine_box(_crop_with_blob(W=120, H=120))
    assert r.success
    # tight box must be smaller than the full crop (blob occupies a sub-region)
    assert (r.x1 - r.x0) < 120
    assert (r.y1 - r.y0) < 120

def test_refine_axis_horizontal():
    r = refine_box(_crop_with_blob(angle_deg=0))
    assert circ_dist(r.axis_deg, 0.0, period=180) < 10.0
    assert r.eccentricity > 0.6

def test_refine_axis_diagonal():
    r = refine_box(_crop_with_blob(angle_deg=45))
    assert circ_dist(r.axis_deg, 45.0, period=180) < 12.0

def test_refine_round_blob_low_eccentricity():
    r = refine_box(_crop_with_blob(sx=10.0, sy=10.0))
    assert r.success
    assert r.eccentricity < 0.45

def test_refine_blank_crop_fails_gracefully():
    blank = np.full((100, 100), 50, dtype=np.uint8)
    r = refine_box(blank)
    assert r.success is False


def test_refine_prefers_centered_blob_over_larger_offcenter():
    # animal = small centered blob; background = a LARGER bright blob in the corner.
    W = H = 140
    img = np.full((H, W), 20, dtype=np.uint8)
    # large off-center (corner) bright region
    yy, xx = np.mgrid[0:H, 0:W]
    corner = np.exp(-(((xx - 15) ** 2 + (yy - 15) ** 2) / (2 * 18.0 ** 2)))
    # smaller centered animal blob
    centre = np.exp(-(((xx - W // 2) ** 2 + (yy - H // 2) ** 2) / (2 * 7.0 ** 2)))
    img = np.clip(20 + 220 * corner + 220 * centre, 0, 255).astype(np.uint8)
    r = refine_box(img)
    assert r.success
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    assert abs(cx - W / 2) < W * 0.25
    assert abs(cy - H / 2) < H * 0.25
