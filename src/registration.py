"""Background registration between two AOS frames.

Each image is an AOS integral; between consecutive frames the drone moves, so the
whole scene shifts. We estimate a 2D similarity transform mapping a point in frame A
to its place in frame B from ORB features on the (cooler, static) background, so we can
later subtract the drone's ego-motion and keep only the animal's residual movement.

Work on a 0.5x grayscale copy for speed and rescale the translation back to full res.
Thermal AOS frames are low-contrast, so we CLAHE them before ORB and match with a
Lowe ratio test (knnMatch) rather than crossCheck -- both lift the inlier ratio a lot.
"""
from __future__ import annotations

import cv2
import numpy as np

_DOWNSCALE = 0.5
_RATIO = 0.75  # Lowe ratio test threshold
_orb = cv2.ORB_create(2000)
_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def estimate_transform(img_a: np.ndarray, img_b: np.ndarray):
    """Affine-partial (similarity) transform A->B from background ORB features.

    Returns (M 2x3 float64 mapping full-res A to full-res B, inlier_ratio) or
    (None, 0.0) if too few matches.
    """
    ga = _clahe.apply(_gray_small(img_a))
    gb = _clahe.apply(_gray_small(img_b))
    ka, da = _orb.detectAndCompute(ga, None)
    kb, db = _orb.detectAndCompute(gb, None)
    if da is None or db is None or len(ka) < 3 or len(kb) < 3:
        return None, 0.0

    matches = _ratio_match(da, db)
    if len(matches) < 3:
        return None, 0.0

    pa = np.float32([ka[m.queryIdx].pt for m in matches])
    pb = np.float32([kb[m.trainIdx].pt for m in matches])
    M, inliers = cv2.estimateAffinePartial2D(
        pa, pb, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return None, 0.0

    inlier_ratio = float(inliers.sum()) / len(inliers) if inliers is not None else 0.0
    return _rescale_to_full(M), inlier_ratio


def _ratio_match(da, db):
    """knnMatch (k=2) kept where the best match beats the second by the Lowe ratio."""
    if len(da) < 2 or len(db) < 2:
        return []
    good = []
    for pair in _matcher.knnMatch(da, db, k=2):
        if len(pair) == 2 and pair[0].distance < _RATIO * pair[1].distance:
            good.append(pair[0])
    return good


def apply_transform(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Map a full-res point (x, y) through the 2x3 transform."""
    nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return float(nx), float(ny)


def _gray_small(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(img, None, fx=_DOWNSCALE, fy=_DOWNSCALE,
                      interpolation=cv2.INTER_AREA)


def _rescale_to_full(M: np.ndarray) -> np.ndarray:
    # M was estimated on downscaled coords; the rotation/scale block is invariant but
    # the translation is in small-image pixels, so scale it back up.
    M = M.astype(np.float64).copy()
    M[:, 2] /= _DOWNSCALE
    return M
