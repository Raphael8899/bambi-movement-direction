"""Tests for centroid association and tracklet construction."""
import numpy as np
import pandas as pd
import pytest

from src.tracking import associate, build_tracklets


def _frame(rows):
    return pd.DataFrame(rows, columns=["cx", "cy", "cls"])


def test_associate_picks_nearest_same_class():
    prev = _frame([(100.0, 100.0, 0)])
    curr = _frame([(300.0, 300.0, 0), (110.0, 105.0, 0)])
    matches = associate(prev, curr, gate_px=200.0)
    assert matches == [(0, 1)]  # the closer detection wins


def test_associate_respects_class():
    prev = _frame([(100.0, 100.0, 0)])
    curr = _frame([(105.0, 105.0, 1), (160.0, 100.0, 0)])
    matches = associate(prev, curr, gate_px=200.0)
    assert matches == [(0, 1)]  # skips the nearer wrong-class detection


def test_associate_gate_excludes_far():
    prev = _frame([(100.0, 100.0, 0)])
    curr = _frame([(900.0, 900.0, 0)])
    assert associate(prev, curr, gate_px=100.0) == []


def test_associate_one_to_one():
    prev = _frame([(100.0, 100.0, 0), (500.0, 500.0, 0)])
    curr = _frame([(110.0, 100.0, 0), (510.0, 500.0, 0)])
    matches = dict(associate(prev, curr, gate_px=200.0))
    assert matches == {0: 0, 1: 1}


def test_build_tracklets_links_moving_animal_across_frames():
    # one animal stepping right by ~40 px each of three frames in one flight
    df = pd.DataFrame({
        "flight_id": [7, 7, 7],
        "frame_num": [10, 11, 12],
        "cls": [0, 0, 0],
        "xc": [0.10, 0.12, 0.14],
        "yc": [0.50, 0.50, 0.50],
    })
    out = build_tracklets(df)
    assert out["tracklet_id"].nunique() == 1


def test_build_tracklets_separates_two_animals():
    df = pd.DataFrame({
        "flight_id": [7, 7, 7, 7],
        "frame_num": [10, 10, 11, 11],
        "cls": [0, 1, 0, 1],
        "xc": [0.10, 0.80, 0.11, 0.81],
        "yc": [0.10, 0.80, 0.10, 0.80],
    })
    out = build_tracklets(df)
    assert out["tracklet_id"].nunique() == 2
    # each class keeps a single tracklet across both frames
    assert out.groupby("cls")["tracklet_id"].nunique().tolist() == [1, 1]


def test_build_tracklets_new_id_when_gap_too_large():
    # animal jumps across the whole image between frames -> not linked
    df = pd.DataFrame({
        "flight_id": [7, 7],
        "frame_num": [10, 11],
        "cls": [0, 0],
        "xc": [0.05, 0.95],
        "yc": [0.05, 0.95],
    })
    out = build_tracklets(df)
    assert out["tracklet_id"].nunique() == 2


def test_build_tracklets_ignores_malformed_flight():
    df = pd.DataFrame({
        "flight_id": [-1, -1],
        "frame_num": [10, 11],
        "cls": [0, 0],
        "xc": [0.10, 0.11],
        "yc": [0.10, 0.10],
    })
    out = build_tracklets(df)
    assert (out["tracklet_id"] == -1).all()
