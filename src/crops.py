"""Crop extraction + simple visibility/elongation features for stratified sampling."""
from __future__ import annotations

import cv2
import numpy as np

from src.data_loader import yolo_to_box
from src.gst import gst_orientation


def extract_crop(image: np.ndarray, xc, yc, w, h, pad: float = 0.3,
                 img_size: int = 2048, square: bool = True):
    """Extract a (square) crop centred on the box with padding. Returns (crop, (x0,y0))."""
    x0, y0, x1, y1 = yolo_to_box(xc, yc, w, h, img_size)
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    if square:
        half = max(bw, bh) * (0.5 + pad)
        hw = hh = half
    else:
        hw, hh = bw * (0.5 + pad), bh * (0.5 + pad)
    X0, Y0 = max(0, int(cx - hw)), max(0, int(cy - hh))
    X1, Y1 = min(img_size, int(cx + hw)), min(img_size, int(cy + hh))
    return image[Y0:Y1, X0:X1], (X0, Y0)


def contrast_to_surround(image: np.ndarray, xc, yc, w, h, img_size: int = 2048) -> float:
    """Mean intensity inside the box minus mean of a surrounding ring (animal visibility)."""
    x0, y0, x1, y1 = (int(v) for v in yolo_to_box(xc, yc, w, h, img_size))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inner = image[y0:y1, x0:x1]
    rx0, ry0 = max(0, x0 - (x1 - x0)), max(0, y0 - (y1 - y0))
    rx1, ry1 = min(img_size, x1 + (x1 - x0)), min(img_size, y1 + (y1 - y0))
    ring = image[ry0:ry1, rx0:rx1]
    if inner.size == 0 or ring.size == 0:
        return 0.0
    return float(inner.mean() - ring.mean())


def crop_coherence(crop_gray: np.ndarray) -> float:
    """GST coherence of a crop (0 round/compact .. 1 elongated/smeared)."""
    return gst_orientation(crop_gray)[1]
