"""Tests for the circular-statistics helpers.

Angles in degrees; period=360 is directional, period=180 is axial.
"""
import numpy as np
import pytest

from src.metrics_circular import (
    circ_dist, circ_mae, circ_median, acc_at,
    circ_mean, resultant_length, rayleigh_p, circ_corr,
)


# --- circ_dist ---------------------------------------------------------------
def test_circ_dist_wraps_across_zero():
    assert circ_dist(10, 350) == pytest.approx(20.0)
    assert circ_dist(350, 10) == pytest.approx(20.0)  # symmetric

def test_circ_dist_max_is_half_period():
    assert circ_dist(0, 180) == pytest.approx(180.0)
    assert circ_dist(0, 200) == pytest.approx(160.0)

def test_circ_dist_identity_is_zero():
    assert circ_dist(123.4, 123.4) == pytest.approx(0.0)

def test_circ_dist_axial_period_180():
    # On the 180-circle, 10 and 170 are 20 apart (not 160).
    assert circ_dist(10, 170, period=180) == pytest.approx(20.0)
    assert circ_dist(0, 90, period=180) == pytest.approx(90.0)
    # A 180-deg flip is zero distance for axial data:
    assert circ_dist(30, 210, period=180) == pytest.approx(0.0)

def test_circ_dist_array_elementwise():
    d = circ_dist(np.array([10, 0, 90]), np.array([350, 5, 270]))
    np.testing.assert_allclose(d, [20, 5, 180])


# --- aggregate error metrics -------------------------------------------------
def test_circ_mae():
    assert circ_mae([10, 0], [350, 5]) == pytest.approx(12.5)

def test_circ_median_robust_to_outlier():
    # three small errors + one catastrophic flip -> median stays small
    pred = [0, 0, 0, 0]
    gt = [2, 4, 6, 180]
    assert circ_median(pred, gt) == pytest.approx(5.0)   # median of [2,4,6,180]
    assert circ_mae(pred, gt) == pytest.approx(48.0)     # mean is wrecked by the flip

def test_acc_at_threshold():
    pred = [10, 100, 0]
    gt = [0, 0, 0]            # dists 10, 100, 0
    assert acc_at(pred, gt, 22.5) == pytest.approx(2/3)
    assert acc_at(pred, gt, 5) == pytest.approx(1/3)

def test_acc_at_axial_flip_counts_as_correct():
    # directional dist 175 but axial it's a 5-deg error -> correct under period=180
    assert acc_at([175], [0], k=10, period=180) == pytest.approx(1.0)
    assert acc_at([175], [0], k=10, period=360) == pytest.approx(0.0)


# --- circular mean / resultant ----------------------------------------------
def test_circ_mean_across_zero():
    assert circ_dist(circ_mean([10, 350]), 0.0) == pytest.approx(0.0, abs=1e-6)

def test_circ_mean_simple():
    assert circ_mean([0, 90]) == pytest.approx(45.0)

def test_circ_mean_in_range():
    m = circ_mean([350, 10, 0])
    assert 0 <= m < 360

def test_resultant_length_concentrated_is_one():
    assert resultant_length([30, 30, 30]) == pytest.approx(1.0)

def test_resultant_length_uniform_is_zero():
    assert resultant_length([0, 90, 180, 270]) == pytest.approx(0.0, abs=1e-9)

def test_resultant_length_axial_uniform():
    # 0 and 90 on the axial (mod 180) circle are opposite -> R-bar = 0
    assert resultant_length([0, 90], period=180) == pytest.approx(0.0, abs=1e-9)
    # but 0 and 180 are the SAME axis -> R-bar = 1
    assert resultant_length([0, 180], period=180) == pytest.approx(1.0)


# --- Rayleigh test -----------------------------------------------------------
def test_rayleigh_concentrated_is_significant():
    rng = np.random.default_rng(0)
    angles = 30 + rng.normal(0, 5, size=50)
    assert rayleigh_p(angles) < 0.01

def test_rayleigh_uniform_not_significant():
    angles = np.array([0, 90, 180, 270], dtype=float)
    assert rayleigh_p(angles) > 0.5


# --- circular correlation ----------------------------------------------------
def test_circ_corr_perfect():
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 360, size=100)
    assert circ_corr(a, a) == pytest.approx(1.0, abs=1e-6)

def test_circ_corr_independent_is_small():
    rng = np.random.default_rng(2)
    a = rng.uniform(0, 360, size=500)
    b = rng.uniform(0, 360, size=500)
    assert abs(circ_corr(a, b)) < 0.2

def test_circ_corr_in_bounds():
    rng = np.random.default_rng(3)
    a = rng.uniform(0, 360, size=50)
    b = rng.uniform(0, 360, size=50)
    assert -1.0 <= circ_corr(a, b) <= 1.0
