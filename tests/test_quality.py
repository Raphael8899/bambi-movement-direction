"""Tests for the ghosting/border-artefact heuristic.

Synthetic cases only: a clean frame with detail in the middle, a frame with a strong
shifted/duplicated border, and a flat frame. We assert the ordering the score is meant
to capture, not absolute thresholds.
"""
import numpy as np
import pytest

from src.quality import ghosting_score, is_ghosted, border_ring_ratio


def _clean_frame(n=256, rng=None):
    """Dark frame with textured detail confined to the centre, quiet border."""
    rng = rng or np.random.default_rng(0)
    img = np.full((n, n), 20, dtype=np.float32)
    c = n // 4
    img[c:n - c, c:n - c] += rng.normal(60, 25, size=(n - 2 * c, n - 2 * c))
    return np.clip(img, 0, 255).astype(np.uint8)


def _ghosted_frame(n=256, shift=8, rng=None):
    """Clean frame plus a bright displaced copy of itself pasted into the border ring."""
    rng = rng or np.random.default_rng(0)
    base = _clean_frame(n, rng).astype(np.float32)
    shifted = np.roll(np.roll(base, shift, axis=0), shift, axis=1)
    out = base.copy()
    b = n // 10
    mask = np.zeros((n, n), bool)
    mask[:b, :] = mask[n - b:, :] = mask[:, :b] = mask[:, n - b:] = True
    out[mask] = np.clip(out[mask] + shifted[mask], 0, 255)
    return out.astype(np.uint8)


def _flat_frame(n=256):
    return np.full((n, n), 30, dtype=np.uint8)


def test_ghosted_scores_higher_than_clean():
    clean = ghosting_score(_clean_frame())
    ghost = ghosting_score(_ghosted_frame())
    assert ghost > clean


def test_flat_frame_low_score():
    # No structure anywhere -> ring ratio falls back to ~0, score stays small.
    assert ghosting_score(_flat_frame()) < 1.0


def test_flat_below_ghosted():
    assert ghosting_score(_flat_frame()) < ghosting_score(_ghosted_frame())


def test_is_ghosted_threshold():
    g = _ghosted_frame()
    s = ghosting_score(g)
    assert is_ghosted(g, thr=s - 0.1)
    assert not is_ghosted(g, thr=s + 0.1)


def test_ring_ratio_grows_with_border_energy():
    rng = np.random.default_rng(1)
    assert border_ring_ratio(_ghosted_frame(rng=rng)) > border_ring_ratio(_clean_frame(rng=rng))


def test_handles_color_input():
    g = _ghosted_frame()
    color = np.stack([g, g, g], axis=-1)
    assert ghosting_score(color) == pytest.approx(ghosting_score(g), rel=1e-6)


def test_empty_input_zero():
    assert ghosting_score(np.zeros((0, 0), np.uint8)) == 0.0
