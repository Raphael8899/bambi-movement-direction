"""Circular statistics for angular predictions, all in degrees.

period=360 is directional data (head/tail distinguished); period=180 is axial data
(a body/blur axis with no head/tail). Errors use the wrapped shortest-arc distance so
the 0/period seam doesn't blow up, and axial data is handled by doubling the angle
before mapping it onto the unit circle.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "circ_dist", "circ_mae", "circ_median", "acc_at",
    "circ_mean", "resultant_length", "rayleigh_p", "circ_corr",
]


def _phase(angles, period):
    """Map angle(deg) to a unit-circle phase(rad) where one ``period`` = full turn."""
    return np.asarray(angles, dtype=float) * (2.0 * np.pi / period)


def _maybe_scalar(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def circ_dist(a, b, period: float = 360.0):
    """Shortest-arc distance in [0, period/2]. Element-wise; scalar in -> float out."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.abs((a - b) % period)
    d = np.minimum(d, period - d)
    return _maybe_scalar(d)


def circ_mae(pred, gt, period: float = 360.0) -> float:
    """Mean absolute angular error."""
    return float(np.mean(circ_dist(pred, gt, period)))


def circ_median(pred, gt, period: float = 360.0) -> float:
    """Median absolute angular error (robust to 180-deg flip outliers)."""
    return float(np.median(circ_dist(pred, gt, period)))


def acc_at(pred, gt, k: float, period: float = 360.0) -> float:
    """Fraction of predictions within k degrees (circular)."""
    return float(np.mean(circ_dist(pred, gt, period) <= k))


def circ_mean(angles, period: float = 360.0) -> float:
    """Circular mean direction in [0, period)."""
    ph = _phase(angles, period)
    m = np.arctan2(np.mean(np.sin(ph)), np.mean(np.cos(ph)))
    deg = m * (period / (2.0 * np.pi))
    out = float(deg % period)
    if out >= period:  # rounding can push a tiny-negative angle up to exactly `period`
        out -= period
    return out


def resultant_length(angles, period: float = 360.0) -> float:
    """Mean resultant length R-bar in [0,1] (0 = uniform, 1 = perfectly concentrated)."""
    ph = _phase(angles, period)
    return float(np.abs(np.mean(np.exp(1j * ph))))


def rayleigh_p(angles, period: float = 360.0) -> float:
    """Rayleigh test of uniformity. Small p -> a significant mean direction exists.

    Uses Zar's approximation for the p-value of Z = n * R-bar^2.
    """
    angles = np.asarray(angles, dtype=float)
    n = angles.size
    if n == 0:
        return 1.0
    R = resultant_length(angles, period)
    Z = n * R * R
    p = np.exp(-Z) * (
        1.0
        + (2.0 * Z - Z * Z) / (4.0 * n)
        - (24.0 * Z - 132.0 * Z**2 + 76.0 * Z**3 - 9.0 * Z**4) / (288.0 * n * n)
    )
    return float(min(1.0, max(0.0, p)))


def circ_corr(a, b, period: float = 360.0) -> float:
    """Jammalamadaka-Sarma circular correlation coefficient in [-1, 1]."""
    pha = _phase(a, period)
    phb = _phase(b, period)
    abar = np.arctan2(np.mean(np.sin(pha)), np.mean(np.cos(pha)))
    bbar = np.arctan2(np.mean(np.sin(phb)), np.mean(np.cos(phb)))
    sa = np.sin(pha - abar)
    sb = np.sin(phb - bbar)
    den = np.sqrt(np.sum(sa**2) * np.sum(sb**2))
    if den == 0:
        return 0.0
    return float(np.sum(sa * sb) / den)
