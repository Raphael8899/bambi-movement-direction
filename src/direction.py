"""Recover each animal's real movement direction from its tracklet.

For every consecutive step in a tracklet we register the two frames on the background
(src.registration), warp the previous centroid forward to cancel drone ego-motion, and
take the residual displacement curr - warp(prev). The per-step heading is the angle of
that residual in image pixel coords (x right, y down), so 0 = east, 90 = south, in
[0,360). Headings are aggregated with the circular mean; concentration (R) and the
Rayleigh p-value say whether the direction is real or just noise.

A direction is TRUSTED only with enough steps, decent registration, a concentrated
heading, and a net displacement above a small noise floor. Otherwise it is recorded as
untrusted -- which for a genuinely stationary animal is the correct outcome.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.metrics_circular import circ_mean, resultant_length, rayleigh_p
from src.registration import apply_transform

MIN_STEPS = 5
MIN_INLIER = 0.5
MIN_R = 0.5
MAX_RAYLEIGH_P = 0.05
NOISE_FLOOR_PX = 5.0  # net displacements below this are within registration noise


@dataclass
class DirectionResult:
    n_steps: int
    mean_dir_deg: float
    R: float
    rayleigh_p: float
    median_inlier: float
    disp_px: float
    trusted: bool


def residual_heading(M, prev_xy, curr_xy):
    """Heading (deg, [0,360)) of curr minus the ego-motion-warped prev point.

    M maps prev frame -> curr frame; if M is None we fall back to the raw step (no
    registration available). x right, y down, so atan2(dy, dx) gives 0=east, 90=south.
    """
    px, py = prev_xy
    cx, cy = curr_xy
    if M is not None:
        px, py = apply_transform(M, px, py)
    dx, dy = cx - px, cy - py
    return math.degrees(math.atan2(dy, dx)) % 360.0, (dx, dy)


def is_trusted(n_steps: int, median_inlier: float, R: float, p: float, disp_px: float,
               min_steps: int = MIN_STEPS, min_inlier: float = MIN_INLIER,
               min_r: float = MIN_R, max_p: float = MAX_RAYLEIGH_P,
               noise_floor: float = NOISE_FLOOR_PX) -> bool:
    return (n_steps >= min_steps and median_inlier >= min_inlier and R >= min_r
            and p < max_p and disp_px >= noise_floor)


def tracklet_direction(headings, residuals, inlier_ratios) -> DirectionResult:
    """Aggregate per-step headings + residuals into a direction with confidence.

    headings: per-step heading in degrees. residuals: per-step (dx, dy) in px (used for
    the net displacement). inlier_ratios: per-step registration inlier ratio.
    """
    n = len(headings)
    if n == 0:
        return DirectionResult(0, float("nan"), 0.0, 1.0, 0.0, 0.0, False)

    mean_dir = circ_mean(headings)
    R = resultant_length(headings)
    p = rayleigh_p(headings)
    median_inlier = float(np.median(inlier_ratios)) if inlier_ratios else 0.0

    res = np.asarray(residuals, dtype=float)
    net = res.sum(axis=0)
    disp_px = float(np.hypot(net[0], net[1]))

    trusted = is_trusted(n, median_inlier, R, p, disp_px)
    return DirectionResult(n, mean_dir, R, p, median_inlier, disp_px, trusted)
