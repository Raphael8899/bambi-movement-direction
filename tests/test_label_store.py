"""Tests for LabelStore: manifest load, setting labels, completeness, resume,
navigation and the saved CSV. Each test builds a small manifest + a few PNGs in a
temp dir."""
import csv
import os

import pytest
from PIL import Image

from src.annotation.label_store import LabelStore

MANIFEST_COLUMNS = ["crop_id", "file", "species", "flight_id", "frame_num", "orig_long_px"]
LABEL_COLUMNS = [
    "crop_id", "motion_state", "direction_class",
    "direction_deg", "annotator", "timestamp_iso",
]


def _make_png(path, size=(64, 48), color=(123, 200, 50)):
    """Create a tiny real PNG so image loading code can be exercised."""
    Image.new("RGB", size, color).save(path)


def _write_manifest(tmp_path, rows):
    """Write a manifest CSV in tmp_path and create the referenced PNGs.

    `rows` is a list of dicts with at least crop_id/file/species. Missing
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
                "species": r.get("species", "deer"),
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
        {"crop_id": f"c{i}", "file": f"c{i}.png", "species": f"sp{i}"}
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
    assert rec["species"] == "sp0"
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
        {"crop_id": f"c{i}", "file": f"c{i}.png", "species": "sp"}
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
    rows = [{"crop_id": "c0", "file": "c0.png", "species": "sp"}]
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
