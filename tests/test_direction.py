"""Tests for residual-heading recovery, trusted gating, and tracklet aggregation."""
import math

import numpy as np
import pytest

from src.direction import (residual_heading, is_trusted, tracklet_direction,
                           NOISE_FLOOR_PX)
from src.registration import apply_transform


def _translation(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)


def test_residual_pure_translation_cancels_ego_motion():
    # the drone shifts everything by (+50, +30); a stationary animal moves with it,
    # so the residual after warping prev forward is ~zero.
    M = _translation(50.0, 30.0)
    prev = (100.0, 100.0)
    curr = (150.0, 130.0)
    _, (dx, dy) = residual_heading(M, prev, curr)
    assert dx == pytest.approx(0.0)
    assert dy == pytest.approx(0.0)


def test_residual_heading_east():
    # no ego-motion, animal steps +x -> heading 0 (east)
    M = _translation(0.0, 0.0)
    h, _ = residual_heading(M, (0.0, 0.0), (10.0, 0.0))
    assert h == pytest.approx(0.0)


def test_residual_heading_south_is_90():
    # y points down, so +y is 90 deg
    M = _translation(0.0, 0.0)
    h, _ = residual_heading(M, (0.0, 0.0), (0.0, 10.0))
    assert h == pytest.approx(90.0)


def test_residual_heading_with_real_motion_on_top_of_drift():
    # drone shifts (+50,+30); animal additionally walks (+8, 0) -> residual heading east
    M = _translation(50.0, 30.0)
    prev = (100.0, 100.0)
    wx, wy = apply_transform(M, *prev)
    curr = (wx + 8.0, wy)
    h, (dx, dy) = residual_heading(M, prev, curr)
    assert dx == pytest.approx(8.0)
    assert dy == pytest.approx(0.0)
    assert h == pytest.approx(0.0)


def test_residual_none_transform_uses_raw_step():
    h, (dx, dy) = residual_heading(None, (0.0, 0.0), (0.0, -10.0))
    assert (dx, dy) == (0.0, -10.0)
    assert h == pytest.approx(270.0)  # -y is north


def test_is_trusted_all_conditions_met():
    assert is_trusted(n_steps=6, median_inlier=0.6, R=0.7, p=0.01, disp_px=30.0)


def test_is_trusted_rejects_few_steps():
    assert not is_trusted(n_steps=4, median_inlier=0.9, R=0.9, p=0.0, disp_px=50.0)


def test_is_trusted_rejects_low_inlier():
    assert not is_trusted(n_steps=8, median_inlier=0.3, R=0.9, p=0.0, disp_px=50.0)


def test_is_trusted_rejects_low_R():
    assert not is_trusted(n_steps=8, median_inlier=0.9, R=0.4, p=0.0, disp_px=50.0)


def test_is_trusted_rejects_high_p():
    assert not is_trusted(n_steps=8, median_inlier=0.9, R=0.9, p=0.2, disp_px=50.0)


def test_is_trusted_rejects_below_noise_floor():
    assert not is_trusted(n_steps=8, median_inlier=0.9, R=0.9, p=0.0,
                          disp_px=NOISE_FLOOR_PX - 0.1)


def test_tracklet_direction_concentrated_eastward_is_trusted():
    # six consistent eastward steps of 6 px each
    headings = [0.0, 358.0, 2.0, 1.0, 359.0, 0.0]
    residuals = [(6.0, 0.0)] * 6
    inliers = [0.7] * 6
    r = tracklet_direction(headings, residuals, inliers)
    assert r.trusted
    assert r.n_steps == 6
    assert min(r.mean_dir_deg, 360 - r.mean_dir_deg) < 5.0
    assert r.R > 0.9
    assert r.disp_px == pytest.approx(36.0)


def test_tracklet_direction_scattered_headings_untrusted():
    headings = [0.0, 90.0, 180.0, 270.0, 45.0, 135.0]
    residuals = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0),
                 (0.7, 0.7), (-0.7, 0.7)]
    inliers = [0.8] * 6
    r = tracklet_direction(headings, residuals, inliers)
    assert not r.trusted
    assert r.R < 0.5


def test_tracklet_direction_empty():
    r = tracklet_direction([], [], [])
    assert r.n_steps == 0
    assert not r.trusted
