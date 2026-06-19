"""Bounding-box refinement: tighten an (oversized, manual) box to the warm animal blob.

The supervisors note the manual boxes "tend to be too large", so before any size
statistic or orientation estimate we segment the warm blob and recompute a tight box,
its body/blur axis (image moments) and an eccentricity/elongation confidence.

Axis convention matches src.gst: image coords, [0,180), 0 = horizontal (+x), 90 = vertical.
Assumes the animal is the BRIGHT (warm) region against a cooler background.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RefinedBB:
    success: bool
    x0: int
    y0: int
    x1: int
    y1: int
    axis_deg: float        # body/blur long-axis [0,180)
    eccentricity: float    # 0 = round, ->1 = elongated
    area_frac: float       # blob area / crop area


def segment_warm_blob(crop_gray: np.ndarray, center=None) -> np.ndarray | None:
    """Otsu-threshold the bright animal, clean morphologically, return the chosen blob.

    The animal sits near the middle of the crop, so we take the largest component whose
    centroid falls in the central 60% (this skips big bright patches of background near
    the edges) and fall back to the largest overall if none do.
    """
    g = cv2.GaussianBlur(crop_gray, (0, 0), 1.0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Animal (bright) should be the foreground minority; invert if Otsu picked background.
    if bw.mean() > 127:
        bw = 255 - bw
    k = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(bw)
    if n <= 1:
        return None
    H, W = bw.shape
    areas = stats[1:, cv2.CC_STAT_AREA]
    cents_fg = cents[1:]
    mx, my = 0.2 * W, 0.2 * H
    central = [i for i in range(len(areas))
               if mx <= cents_fg[i, 0] <= W - mx and my <= cents_fg[i, 1] <= H - my]
    if central:
        best = max(central, key=lambda i: areas[i])
    else:
        best = int(np.argmax(areas))
    return (lbl == best + 1).astype(np.uint8) * 255


def _moment_axis_ecc(mask: np.ndarray) -> tuple[float, float]:
    """Body axis (deg, [0,180)) and eccentricity from second-order central moments."""
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] == 0:
        return 0.0, 0.0
    mu20 = M["mu20"] / M["m00"]
    mu02 = M["mu02"] / M["m00"]
    mu11 = M["mu11"] / M["m00"]
    theta = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)  # major-axis angle, image coords
    axis_deg = float(np.degrees(theta) % 180.0)
    common = np.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11 ** 2)
    lam1 = (mu20 + mu02 + common) / 2.0
    lam2 = (mu20 + mu02 - common) / 2.0
    ecc = float(np.sqrt(1.0 - lam2 / lam1)) if lam1 > 0 else 0.0
    return axis_deg, ecc


def refine_box(crop_gray: np.ndarray) -> RefinedBB:
    """Refine one crop. Returns RefinedBB; success=False if no blob found."""
    H, W = crop_gray.shape[:2]
    mask = segment_warm_blob(crop_gray)
    if mask is None or int(mask.sum()) == 0:
        return RefinedBB(False, 0, 0, W, H, 0.0, 0.0, 1.0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return RefinedBB(False, 0, 0, W, H, 0.0, 0.0, 1.0)
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    area_frac = float(cv2.contourArea(cnt) / (W * H))
    axis_deg, ecc = _moment_axis_ecc(mask)
    return RefinedBB(True, int(x), int(y), int(x + w), int(y + h), axis_deg, ecc, area_frac)
