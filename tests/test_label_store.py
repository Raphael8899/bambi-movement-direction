"""Tests for LabelStore: manifest load, setting labels, completeness, resume,
navigation and the saved CSV. Each test builds a small manifest + a few PNGs in a
temp dir."""
import csv
import os

import pytest
from PIL import Image

from src.annotation.label_store import LabelStore

MANIFEST_COLUMNS = ["crop_id", "file", "class_id", "flight_id", "frame_num", "orig_long_px"]
LABEL_COLUMNS = [
    "crop_id", "motion_state", "direction_class",
    "direction_deg", "annotator", "timestamp_iso",
]


def _make_png(path, size=(64, 48), color=(123, 200, 50)):
    """Create a tiny real PNG so image loading code can be exercised."""
    Image.new("RGB", size, color).save(path)


def _write_manifest(tmp_path, rows):
    """Write a manifest CSV in tmp_path and create the referenced PNGs.

    `rows` is a list of dicts with at least crop_id/file/class_id. Missing
    columns are filled with placeholder values.
    """
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for i, r in enumerate(rows):
            row = {
                "crop_id": r["crop_id"],
                "file": r["file"],
                "class_id": r.get("class_id", "0"),
                "flight_id": r.get("flight_id", "F1"),
                "frame_num": r.get("frame_num", i),
                "orig_long_px": r.get("orig_long_px", 100),
            }
            w.writerow(row)
            # create the PNG referenced by this row (unless told not to)
            if r.get("make_png", True):
                _make_png(tmp_path / r["file"])
    return manifest


def _basic_store(tmp_path, n=3):
    """A store with `n` crops named c0..c(n-1)."""
    rows = [
        {"crop_id": f"c{i}", "file": f"c{i}.png", "class_id": f"sp{i}"}
        for i in range(n)
    ]
    manifest = _write_manifest(tmp_path, rows)
    labels_out = tmp_path / "labels.csv"
    return LabelStore(str(manifest), str(labels_out), annotator="tester")


# --- manifest load ----------------------------------------------------------

def test_manifest_load_count_and_fields(tmp_path):
    store = _basic_store(tmp_path, n=3)
    assert store.count() == 3
    rec = store.record(0)
    assert rec["crop_id"] == "c0"
    assert rec["file"] == "c0.png"
    assert rec["class_id"] == "sp0"
    # file path should resolve to the manifest's directory
    assert os.path.basename(store.image_path(0)) == "c0.png"
    assert os.path.dirname(store.image_path(0)) == os.path.dirname(str(store.manifest_path))


def test_manifest_load_starts_at_index_zero_when_no_labels(tmp_path):
    store = _basic_store(tmp_path, n=3)
    assert store.index == 0


# --- set_motion / set_direction / is_complete -------------------------------

def test_set_motion_and_direction(tmp_path):
    store = _basic_store(tmp_path, n=2)
    store.set_motion("moving")
    store.set_direction(2)  # E
    lbl = store.label(0)
    assert lbl["motion_state"] == "moving"
    assert lbl["direction_class"] == 2


def test_is_complete_requires_both(tmp_path):
    store = _basic_store(tmp_path, n=2)
    assert store.is_complete(0) is False
    store.set_motion("stationary")
    assert store.is_complete(0) is False  # direction still missing
    store.set_direction(0)  # N
    assert store.is_complete(0) is True


def test_is_complete_false_for_untouched(tmp_path):
    store = _basic_store(tmp_path, n=2)
    assert store.is_complete(1) is False


def test_set_direction_none_autocompletes(tmp_path):
    """Pressing 0 (none) should make the crop complete on its own."""
    store = _basic_store(tmp_path, n=2)
    store.set_none()  # convenience for the '0' key
    assert store.is_complete(0) is True
    lbl = store.label(0)
    assert lbl["direction_class"] == -2
    assert lbl["motion_state"] == "unsure"


def test_invalid_motion_rejected(tmp_path):
    store = _basic_store(tmp_path, n=1)
    with pytest.raises(ValueError):
        store.set_motion("flying")


def test_invalid_direction_rejected(tmp_path):
    store = _basic_store(tmp_path, n=1)
    with pytest.raises(ValueError):
        store.set_direction(99)


# --- direction_class_to_deg -------------------------------------------------

@pytest.mark.parametrize("cls,deg", [
    (0, 0.0),     # N
    (1, 45.0),    # NE
    (2, 90.0),    # E
    (3, 135.0),   # SE
    (4, 180.0),   # S
    (5, 225.0),   # SW
    (6, 270.0),   # W
    (7, 315.0),   # NW
])
def test_direction_class_to_deg_compass(cls, deg):
    assert LabelStore.direction_class_to_deg(cls) == deg


def test_direction_class_to_deg_special_are_empty(tmp_path):
    # axis-only (-1) and none (-2) -> empty string
    assert LabelStore.direction_class_to_deg(-1) == ""
    assert LabelStore.direction_class_to_deg(-2) == ""


# --- save() -----------------------------------------------------------------

def test_save_writes_correct_columns_and_values(tmp_path):
    store = _basic_store(tmp_path, n=2)
    store.set_motion("moving")
    store.set_direction(2)  # E -> 90.0
    store.save()

    out = tmp_path / "labels.csv"
    assert out.exists()
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        header = csv.DictReader(open(out, encoding="utf-8")).fieldnames

    assert header == LABEL_COLUMNS
    # only completed crops are persisted; c0 is complete, c1 is not
    by_id = {r["crop_id"]: r for r in rows}
    assert "c0" in by_id
    r0 = by_id["c0"]
    assert r0["motion_state"] == "moving"
    assert r0["direction_class"] == "2"
    assert r0["direction_deg"] == "90.0"
    assert r0["annotator"] == "tester"
    assert r0["timestamp_iso"]  # non-empty ISO timestamp


def test_save_special_direction_deg_is_empty(tmp_path):
    store = _basic_store(tmp_path, n=1)
    store.set_motion("unsure")
    store.set_direction(-1)  # axis-only
    store.save()
    out = tmp_path / "labels.csv"
    with open(out, newline="", encoding="utf-8") as f:
        r = list(csv.DictReader(f))[0]
    assert r["direction_class"] == "-1"
    assert r["direction_deg"] == ""  # empty for axis-only


# --- resume -----------------------------------------------------------------

def test_resume_jumps_to_first_unlabeled(tmp_path):
    rows = [
        {"crop_id": f"c{i}", "file": f"c{i}.png", "class_id": "sp"}
        for i in range(4)
    ]
    manifest = _write_manifest(tmp_path, rows)
    labels_out = tmp_path / "labels.csv"
    # Pre-populate labels for c0 and c1 (complete).
    with open(labels_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        w.writerow({"crop_id": "c0", "motion_state": "moving",
                    "direction_class": "2", "direction_deg": "90.0",
                    "annotator": "x", "timestamp_iso": "2026-01-01T00:00:00"})
        w.writerow({"crop_id": "c1", "motion_state": "stationary",
                    "direction_class": "0", "direction_deg": "0.0",
                    "annotator": "x", "timestamp_iso": "2026-01-01T00:00:00"})

    store = LabelStore(str(manifest), str(labels_out), annotator="tester")
    # resume_index points to first unlabeled crop = c2 (index 2)
    assert store.resume_index() == 2
    assert store.index == 2
    # the pre-existing labels are loaded and editable
    assert store.is_complete(0) is True
    assert store.label(0)["motion_state"] == "moving"
    assert store.label(1)["direction_class"] == 0


def test_resume_all_labeled_clamps_to_last(tmp_path):
    rows = [{"crop_id": "c0", "file": "c0.png", "class_id": "sp"}]
    manifest = _write_manifest(tmp_path, rows)
    labels_out = tmp_path / "labels.csv"
    with open(labels_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        w.writeheader()
        w.writerow({"crop_id": "c0", "motion_state": "moving",
                    "direction_class": "2", "direction_deg": "90.0",
                    "annotator": "x", "timestamp_iso": "2026-01-01T00:00:00"})
    store = LabelStore(str(manifest), str(labels_out), annotator="tester")
    # all labeled -> stay on last index (don't go out of bounds)
    assert store.resume_index() == 0


def test_resume_no_labels_file_is_index_zero(tmp_path):
    store = _basic_store(tmp_path, n=3)
    assert store.resume_index() == 0


# --- advance() / back() bounds ---------------------------------------------

def test_advance_increments_index(tmp_path):
    store = _basic_store(tmp_path, n=3)
    assert store.index == 0
    store.advance()
    assert store.index == 1
    store.advance()
    assert store.index == 2


def test_advance_does_not_exceed_last(tmp_path):
    store = _basic_store(tmp_path, n=2)
    store.advance()
    store.advance()  # at last index now
    store.advance()  # should clamp
    assert store.index == 1


def test_back_decrements_and_clamps_at_zero(tmp_path):
    store = _basic_store(tmp_path, n=3)
    store.advance()
    store.advance()
    assert store.index == 2
    store.back()
    assert store.index == 1
    store.back()
    store.back()  # clamp at 0
    assert store.index == 0


def test_completed_count_and_progress(tmp_path):
    store = _basic_store(tmp_path, n=3)
    assert store.completed_count() == 0
    store.set_motion("moving")
    store.set_direction(0)
    assert store.completed_count() == 1
    store.advance()
    store.set_none()
    assert store.completed_count() == 2


# --- direction: axis orientation (key 5) + optional arrowhead at either end --

def test_orientation_and_head_codes():
    from src.annotation.label_store import (axis_index, head_of, make_dir,
                                            AXIS_ONLY, HEAD_FWD, HEAD_REV, N_ORIENT)
    assert N_ORIENT == 8
    for i in range(N_ORIENT):
        assert axis_index(AXIS_ONLY + i) == i and head_of(AXIS_ONLY + i) == "none"
        assert axis_index(HEAD_FWD + i) == i and head_of(HEAD_FWD + i) == "fwd"
        assert axis_index(HEAD_REV + i) == i and head_of(HEAD_REV + i) == "rev"
    assert make_dir(3, "none") == AXIS_ONLY + 3
    assert make_dir(3, "fwd") == HEAD_FWD + 3
    assert make_dir(3, "rev") == HEAD_REV + 3
    assert axis_index(0) is None and axis_index(None) is None   # legacy compass / unset


def test_direction_degrees_axis_vs_arrow():
    from src.annotation.label_store import make_dir
    # orientation i=2 -> a 45 deg line
    assert LabelStore.direction_class_to_deg(make_dir(2, "none")) == 45.0    # axis (axial)
    assert LabelStore.direction_class_to_deg(make_dir(2, "fwd")) == 45.0     # arrow, one end
    assert LabelStore.direction_class_to_deg(make_dir(2, "rev")) == 225.0    # arrow, other end
    assert LabelStore.direction_class_to_deg(make_dir(0, "rev")) == 180.0


def test_rotate_orientation_keeps_head_and_wraps():
    from src.annotation.label_store import rotate_orientation, make_dir, axis_index, head_of
    assert rotate_orientation(None) == make_dir(0, "none")     # first press -> axis only
    r = rotate_orientation(make_dir(7, "fwd"))
    assert axis_index(r) == 0 and head_of(r) == "fwd"          # wraps, keeps the arrow


def test_set_head_keeps_orientation():
    from src.annotation.label_store import set_head, make_dir, axis_index
    c = make_dir(3, "none")
    assert set_head(c, "fwd") == make_dir(3, "fwd")
    assert set_head(c, "rev") == make_dir(3, "rev")
    assert set_head(make_dir(3, "fwd"), "none") == make_dir(3, "none")
    assert axis_index(set_head(None, "fwd")) == 0              # no orientation yet -> first


def test_new_codes_accepted_invalid_rejected(tmp_path):
    from src.annotation.label_store import make_dir
    store = _basic_store(tmp_path, n=1)
    store.set_motion("moving")
    store.set_direction(make_dir(5, "rev"))
    assert store.is_complete(0)
    for bad in (8, 9, 18, 28, 38, -3):
        with pytest.raises(ValueError):
            store.set_direction(bad)


def test_direction_names_present():
    from src.annotation.label_store import COMPASS_NAMES, AXIS_ONLY, HEAD_FWD, HEAD_REV
    assert "axis" in COMPASS_NAMES[AXIS_ONLY].lower()
    assert "head" in COMPASS_NAMES[HEAD_FWD].lower()
    assert "head" in COMPASS_NAMES[HEAD_REV].lower()


def test_save_writes_arrow_heading(tmp_path):
    from src.annotation.label_store import make_dir
    store = _basic_store(tmp_path, n=1)
    store.set_motion("moving")
    store.set_direction(make_dir(2, "rev"))   # 225 deg heading
    store.save()
    r = list(csv.DictReader(open(tmp_path / "labels.csv", encoding="utf-8")))[0]
    assert r["direction_deg"] == "225.0"


def test_slight_motion_state_accepted(tmp_path):
    store = _basic_store(tmp_path, n=1)
    store.set_motion("slight")          # the new in-between of stationary and moving
    assert store.label(0)["motion_state"] == "slight"
