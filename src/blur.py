"""Single-image motion-blur axis estimators for the thermal crops.

A moving animal smears along its direction of travel. Three classical cues recover that
axis from one frame: the gradient orientation histogram (gradients survive across the
smear, not along it), the FFT power spectrum (linear blur lays down dark ripples
perpendicular to the motion), and the cepstrum (a periodic blur kernel leaves an
off-centre peak at the motion offset). All return an axis in [0,180), image coords,
0 = horizontal (+x), 90 = vertical, plus a confidence in [0,1]. These complement
src.gst (structure tensor) and src.bb_refinement (moment axis).
"""
from __future__ import annotations

import cv2
import numpy as np


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """Normalised linear motion-blur PSF of the given length and angle (image coords)."""
    length = max(1, int(round(length)))
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), -angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    s = k.sum()
    return k / s if s > 0 else k


def _otsu_mask(gray: np.ndarray) -> np.ndarray | None:
    g = cv2.GaussianBlur(gray.astype(np.uint8), (0, 0), 1.0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = 255 - bw
    return bw.astype(bool) if bw.any() else None


def blur_axis_gradient(gray: np.ndarray, bins: int = 180,
                       mask: np.ndarray | None = None):
    """Magnitude-weighted gradient-orientation histogram.

    Blur suppresses gradients along the motion and keeps them across it, so the dominant
    gradient direction is perpendicular to the smear; the axis is that peak minus 90 deg.
    Confidence is the histogram's resultant length (axial concentration).
    """
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    ori = (np.degrees(np.arctan2(gy, gx))) % 180.0  # axial gradient direction

    if mask is not None:
        m = mask.astype(bool)
        mag, ori = mag[m], ori[m]
    if mag.sum() <= 0:
        return 0.0, 0.0

    hist, edges = np.histogram(ori, bins=bins, range=(0.0, 180.0), weights=mag)
    centres = 0.5 * (edges[:-1] + edges[1:])
    peak_grad = float(centres[int(np.argmax(hist))])
    axis = (peak_grad - 90.0) % 180.0

    # Axial resultant length of the weighted histogram as a concentration score.
    ph = np.deg2rad(centres) * 2.0
    w = hist / hist.sum()
    conf = float(np.abs(np.sum(w * np.exp(1j * ph))))
    return axis, conf


def _log_power_spectrum(gray: np.ndarray, pad: int = 256) -> np.ndarray:
    """Hann-windowed, zero-padded, fftshifted log power spectrum."""
    g = gray.astype(np.float32)
    g = g - g.mean()
    h, w = g.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    g = g * (wy * wx)
    n = max(pad, h, w)
    F = np.fft.fft2(g, s=(n, n))
    P = np.abs(np.fft.fftshift(F)) ** 2
    return np.log1p(P)


def blur_axis_spectrum(gray: np.ndarray, n_angles: int = 180):
    """Orientation of the spectral ripples via a simple Radon projection.

    The log power spectrum is high-passed (subtract a blurred copy) to expose the ripple
    pattern, then projected at every angle by rotating and summing columns. The motion
    smear lays down dark ripples perpendicular to itself, so summing across them (along
    the motion) gives the most variable projection; that projection direction is the axis.
    """
    S = _log_power_spectrum(gray)
    S = S - cv2.GaussianBlur(S, (0, 0), 3.0)  # keep the oriented ripples

    n = S.shape[0]
    c = (n / 2 - 0.5, n / 2 - 0.5)
    # Limit to a central disc so the rotated corners don't bias column sums.
    yy, xx = np.mgrid[0:n, 0:n]
    r = min(c) * 0.95
    disc = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2) <= r * r
    S = np.where(disc, S, 0.0).astype(np.float32)

    angles = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    scores = np.empty(n_angles)
    for i, a in enumerate(angles):
        M = cv2.getRotationMatrix2D(c, a, 1.0)
        rot = cv2.warpAffine(S, M, (n, n), flags=cv2.INTER_LINEAR)
        proj = rot.sum(axis=0)            # column sums = Radon projection at angle a
        scores[i] = proj.var()

    best = int(np.argmax(scores))
    axis = float(angles[best] % 180.0)

    mean, std = scores.mean(), scores.std()
    conf = float((scores[best] - mean) / std) if std > 0 else 0.0
    conf = float(np.tanh(max(0.0, conf) / 3.0))  # squash peak prominence into [0,1)
    return axis, conf


def blur_axis_cepstrum(gray: np.ndarray, dc_radius: int = 4):
    """Off-centre cepstral peak: its direction is the axis, its radius the blur length.

    cepstrum = ifft(log(|fft|+eps)). A linear blur leaves a symmetric peak pair offset
    from the origin along the motion direction at a radius equal to the blur length. We
    mask the DC neighbourhood and read the strongest remaining peak.
    """
    g = gray.astype(np.float32)
    g = g - g.mean()
    h, w = g.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    g = g * (wy * wx)

    F = np.fft.fft2(g)
    logmag = np.log(np.abs(F) + 1e-6)
    cep = np.fft.fftshift(np.abs(np.fft.ifft2(logmag)))

    n_h, n_w = cep.shape
    cy, cx = n_h // 2, n_w // 2
    yy, xx = np.mgrid[0:n_h, 0:n_w]
    rr = np.hypot(yy - cy, xx - cx)
    # Only the lower-half-plane to avoid double-counting the symmetric peak pair.
    valid = (rr > dc_radius) & (rr < min(cy, cx)) & (yy >= cy)
    if not valid.any():
        return 0.0, 0.0, 0.0

    masked = np.where(valid, cep, -np.inf)
    py, px = np.unravel_index(int(np.argmax(masked)), cep.shape)
    dy, dx = py - cy, px - cx
    axis = (np.degrees(np.arctan2(dy, dx))) % 180.0
    length = float(np.hypot(dy, dx))

    peak = cep[py, px]
    bg = cep[valid]
    med, mad = np.median(bg), np.median(np.abs(bg - np.median(bg))) + 1e-9
    conf = float(np.tanh(max(0.0, (peak - med) / (6.0 * mad))))
    return axis, conf, length
