"""Tests for filename parsing and YOLO geometry (pure logic)."""
import pytest
from src.data_loader import parse_flight_frame, yolo_to_box, long_side_px


def test_parse_basic():
    assert parse_flight_frame("0_8083_jpg.rf.c7ba7713.jpg") == (0, 8083)

def test_parse_multidigit_flight():
    assert parse_flight_frame("212_8500_jpg.rf.deadbeef.jpg") == (212, 8500)

def test_parse_malformed_returns_minus_one():
    assert parse_flight_frame("not_a_valid_name.jpg") == (-1, -1)
    assert parse_flight_frame("garbage.jpg") == (-1, -1)

def test_yolo_to_box_center():
    x0, y0, x1, y1 = yolo_to_box(0.5, 0.5, 0.1, 0.2, img_size=2048)
    assert (x0, y0, x1, y1) == pytest.approx((921.6, 819.2, 1126.4, 1228.8))

def test_yolo_to_box_clips_to_image():
    # box partly outside -> clipped to [0, img]
    x0, y0, x1, y1 = yolo_to_box(0.01, 0.01, 0.1, 0.1, img_size=2048)
    assert x0 == 0.0 and y0 == 0.0
    assert x1 > 0 and y1 > 0

def test_long_side_px():
    assert long_side_px(0.03, 0.02, img_size=2048) == pytest.approx(61.44)
