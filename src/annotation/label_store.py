"""Manifest + label bookkeeping for the annotation tool (no UI here, so it can be tested).

Direction classes: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7 (head visible).
Axis-only orientations (line visible, head/tail unknown): 8 axes every 22.5 deg, classes 10..17
(10 = 0 deg = vertical N-S, 12 = 45 deg, 14 = 90 deg = horizontal, ... 17 = 157.5 deg).
Legacy generic axis-only=-1, none=-2. Headings are in image degrees, 0=up(N), increasing clockwise;
axis degrees are the same convention reduced to [0,180).
"""
import csv
import os
import tempfile
from datetime import datetime, timezone

MANIFEST_COLUMNS = ["crop_id", "file", "class_id", "flight_id", "frame_num", "orig_long_px"]
LABEL_COLUMNS = ["crop_id", "motion_state", "direction_class", "direction_deg",
                 "annotator", "timestamp_iso"]

MOTION_STATES = ("stationary", "slight", "moving", "unsure")

DIR_AXIS_ONLY = -1
DIR_NONE = -2

_COMPASS_DEG = {0: 0.0, 1: 45.0, 2: 90.0, 3: 135.0, 4: 180.0, 5: 225.0, 6: 270.0, 7: 315.0}

# Axis-only orientations: the body/blur line is visible but head vs tail is not.
# Eight axes every 22.5 deg in [0,180), same compass convention (10 = vertical N-S).
AXIS_STEP = 22.5
AXIS_CYCLE = list(range(10, 18))                       # classes 10..17, rotated by repeated presses
AXIS_DEG = {c: round((c - 10) * AXIS_STEP, 1) for c in AXIS_CYCLE}

COMPASS_NAMES = {0: "N", 1: "NE", 2: "E", 3: "SE", 4: "S", 5: "SW", 6: "W", 7: "NW",
                 DIR_AXIS_ONLY: "axis-only", DIR_NONE: "none"}
COMPASS_NAMES.update({c: f"axis {AXIS_DEG[c]:g}°" for c in AXIS_CYCLE})


def _valid_direction(cls):
    return cls in _COMPASS_DEG or cls in AXIS_DEG or cls in (DIR_AXIS_ONLY, DIR_NONE)


def next_axis_class(cls):
    """Next axis orientation for the UI's repeat-press cycle; a non-axis maps to the first axis."""
    if cls in AXIS_CYCLE:
        return AXIS_CYCLE[(AXIS_CYCLE.index(cls) + 1) % len(AXIS_CYCLE)]
    return AXIS_CYCLE[0]


class LabelStore:
    """Loads the manifest, keeps the in-progress labels, and writes them to CSV.

    The output CSV is rewritten after every change and re-read on startup, so the
    annotator can quit and pick up where they left off.
    """

    def __init__(self, manifest_path, labels_path, annotator="andreas"):
        self.manifest_path = manifest_path
        self.labels_path = labels_path
        self.annotator = annotator
        self._base_dir = os.path.dirname(os.path.abspath(manifest_path))

        with open(manifest_path, newline="", encoding="utf-8") as f:
            self._records = [dict(r) for r in csv.DictReader(f)]

        self._labels = {}              # index -> {motion_state, direction_class, timestamp_iso}
        self._read_existing(labels_path)
        self.index = self.resume_index()

    def _read_existing(self, labels_path):
        if not labels_path or not os.path.exists(labels_path):
            return
        id_to_idx = {r["crop_id"]: i for i, r in enumerate(self._records)}
        try:
            with open(labels_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    idx = id_to_idx.get(row.get("crop_id"))
                    if idx is None:
                        continue
                    entry = {}
                    motion = (row.get("motion_state") or "").strip()
                    if motion in MOTION_STATES:
                        entry["motion_state"] = motion
                    raw = (row.get("direction_class") or "").strip()
                    if raw:
                        try:
                            cls = int(raw)
                            if _valid_direction(cls):
                                entry["direction_class"] = cls
                        except ValueError:
                            pass
                    ts = (row.get("timestamp_iso") or "").strip()
                    if ts:
                        entry["timestamp_iso"] = ts
                    if entry:
                        self._labels[idx] = entry
        except (OSError, csv.Error):
            self._labels = {}

    # manifest access
    def count(self):
        return len(self._records)

    def record(self, idx):
        return self._records[idx]

    def crop_id(self, idx=None):
        return self._records[self._i(idx)]["crop_id"]

    def class_id(self, idx=None):
        rec = self._records[self._i(idx)]
        return rec.get("class_id", rec.get("species", ""))  # tolerate old "species"-named manifests

    def image_path(self, idx=None):
        return os.path.join(self._base_dir, self._records[self._i(idx)]["file"])

    def _i(self, idx):
        return self.index if idx is None else idx

    # labels
    def label(self, idx=None):
        raw = self._labels.get(self._i(idx), {})
        return {"motion_state": raw.get("motion_state"),
                "direction_class": raw.get("direction_class")}

    def set_motion(self, state, idx=None):
        if state not in MOTION_STATES:
            raise ValueError(f"unknown motion state: {state!r}")
        i = self._i(idx)
        self._labels.setdefault(i, {})["motion_state"] = state
        self._stamp(i)

    def set_direction(self, cls, idx=None):
        if not _valid_direction(cls):
            raise ValueError(f"unknown direction class: {cls!r}")
        i = self._i(idx)
        self._labels.setdefault(i, {})["direction_class"] = cls
        self._stamp(i)

    def set_none(self, idx=None):
        # the "0" key: nothing usable in this crop
        i = self._i(idx)
        entry = self._labels.setdefault(i, {})
        entry["direction_class"] = DIR_NONE
        entry["motion_state"] = "unsure"
        self._stamp(i)

    def _stamp(self, idx):
        self._labels[idx]["timestamp_iso"] = datetime.now(timezone.utc).isoformat()

    def is_complete(self, idx=None):
        raw = self._labels.get(self._i(idx), {})
        return "motion_state" in raw and "direction_class" in raw

    def missing(self, idx=None):
        raw = self._labels.get(self._i(idx), {})
        out = []
        if "motion_state" not in raw:
            out.append("motion")
        if "direction_class" not in raw:
            out.append("direction")
        return out

    def completed_count(self):
        return sum(1 for i in range(self.count()) if self.is_complete(i))

    # navigation
    def advance(self):
        if self.index < self.count() - 1:
            self.index += 1
        return self.index

    def back(self):
        if self.index > 0:
            self.index -= 1
        return self.index

    def resume_index(self):
        for i in range(self.count()):
            if not self.is_complete(i):
                return i
        return max(0, self.count() - 1)

    @staticmethod
    def direction_class_to_deg(cls):
        if cls in AXIS_DEG:
            return AXIS_DEG[cls]
        return _COMPASS_DEG.get(cls, "")

    def save(self):
        out_dir = os.path.dirname(os.path.abspath(self.labels_path)) or "."
        os.makedirs(out_dir, exist_ok=True)
        # write to a temp file first so an interrupted save can't truncate the real one
        fd, tmp = tempfile.mkstemp(dir=out_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
                w.writeheader()
                for i in range(self.count()):
                    if self.is_complete(i):
                        w.writerow(self._row(i))
            os.replace(tmp, self.labels_path)
        except BaseException:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise

    def _row(self, idx):
        raw = self._labels[idx]
        cls = raw["direction_class"]
        return {"crop_id": self._records[idx]["crop_id"],
                "motion_state": raw["motion_state"],
                "direction_class": cls,
                "direction_deg": self.direction_class_to_deg(cls),
                "annotator": self.annotator,
                "timestamp_iso": raw.get("timestamp_iso",
                                         datetime.now(timezone.utc).isoformat())}
