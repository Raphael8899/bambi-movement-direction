"""A per-image ghosting/border-artefact heuristic for the AOS thermal exports.

Some exports show a sheared or duplicated multi-frame border: the integration leaves a
bright displaced copy of the scene around the edges, or a directional smear that runs
along one frame side. The score below compares structured high-frequency energy in the
outer border ring against the centre, plus the strongest directional gradient streak at
the four edges. Higher = more artefact. This is a heuristic for triage, not a validated
detector - eyeball the montage from scripts/eda or the ghosting examples before trusting
a threshold.

Works on a single grayscale image (any size). Returns plain floats.
"""
from __future__ import annotations

import cv2
import numpy as np


def _grad_energy(g: np.ndarray) -> np.ndarray:
    """Per-pixel gradient magnitude (float32)."""
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def border_ring_ratio(gray: np.ndarray, frac: float = 0.10) -> float:
    """Gradient energy in the outer ring divided by energy in the centre.

    A clean frame has its detail in the middle (animals, ground texture) and a quiet
    border, so the ratio is small. Ghosting dumps a displaced scene copy into the border
    and pushes the ratio up.
    """
    g = gray.astype(np.float32)
    H, W = g.shape[:2]
    bh, bw = max(1, int(H * frac)), max(1, int(W * frac))
    e = _grad_energy(g)

    ring = e.copy()
    ring[bh:H - bh, bw:W - bw] = np.nan
    centre = e[bh:H - bh, bw:W - bw]
    ring_mean = float(np.nanmean(ring))
    centre_mean = float(np.mean(centre)) if centre.size else 0.0
    if centre_mean <= 1e-6:
        return 0.0
    return ring_mean / centre_mean


def edge_streak(gray: np.ndarray, frac: float = 0.06) -> float:
    """Strongest directional smear along the four edge strips.

    A sheared border runs as a long line parallel to the frame edge: on the top/bottom
    strips that is strong horizontal structure (column gradients dominate), on the
    left/right strips strong vertical structure. We take the worst edge and normalise by
    the image's overall gradient so flat frames don't score high.
    """
    g = gray.astype(np.float32)
    H, W = g.shape[:2]
    bh, bw = max(2, int(H * frac)), max(2, int(W * frac))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)

    base = float(np.hypot(gx, gy).mean()) + 1e-6
    strips = [
        np.abs(gy[:bh, :]).mean(),          # top: horizontal streak -> vertical gradient
        np.abs(gy[H - bh:, :]).mean(),      # bottom
        np.abs(gx[:, :bw]).mean(),          # left: vertical streak -> horizontal gradient
        np.abs(gx[:, W - bw:]).mean(),      # right
    ]
    return float(max(strips) / base)


def ghosting_score(gray: np.ndarray) -> float:
    """Combined border-artefact score, higher = more likely ghosted.

    Sum of the border ring ratio and a softened edge-streak term. Both are scale-free
    ratios so the magnitudes are roughly comparable across frames.
    """
    if gray is None or gray.size == 0:
        return 0.0
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    ring = border_ring_ratio(gray)
    streak = edge_streak(gray)
    return float(ring + 0.5 * streak)


def is_ghosted(gray: np.ndarray, thr: float = 2.0) -> bool:
    """True if the ghosting score is at or above thr."""
    return ghosting_score(gray) >= thr
