"""Gradient structure tensor: the dominant orientation of a crop plus a coherence score.

For an elongated warm blob the orientation is the body axis (stationary) or the
motion-blur axis (moving); coherence is near 1 for a clear axis and near 0 for a round
blob. It works in the spatial domain, which holds up better than FFT-based methods on
the small (~40-115 px) thermal crops. Angle in image coords, [0,180): 0 = horizontal,
90 = vertical. (Following the OpenCV anisotropic-segmentation tutorial.)
"""
from __future__ import annotations

import cv2
import numpy as np


def gst_orientation(gray: np.ndarray, sigma: float = 2.0, mask: np.ndarray | None = None):
    """Return (orientation_deg in [0,180), coherence in [0,1]) for a grayscale crop."""
    g = gray.astype(np.float32)
    Ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = cv2.GaussianBlur(Ix * Ix, (0, 0), sigma)
    Jyy = cv2.GaussianBlur(Iy * Iy, (0, 0), sigma)
    Jxy = cv2.GaussianBlur(Ix * Iy, (0, 0), sigma)

    if mask is not None:
        m = mask.astype(bool)
        jxx, jyy, jxy = Jxx[m].sum(), Jyy[m].sum(), Jxy[m].sum()
    else:
        jxx, jyy, jxy = Jxx.sum(), Jyy.sum(), Jxy.sum()

    # Dominant gradient orientation; the structure long-axis is perpendicular to it.
    theta_grad = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    long_axis = (np.degrees(theta_grad) + 90.0) % 180.0

    denom = jxx + jyy
    coh = float(np.sqrt((jxx - jyy) ** 2 + (2.0 * jxy) ** 2) / denom) if denom > 0 else 0.0
    return float(long_axis), coh
