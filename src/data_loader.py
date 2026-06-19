"""Load the YOLO labels into a tidy DataFrame and pull the flight/frame ids out of the
filenames (which look like {flight_id}_{frame_num}_jpg.rf.{hash}.jpg)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from config import DATASET_DIR, SPLITS, IMG_SIZE, CLASS_NAMES
except Exception:  # allow import without project config (tests of pure fns)
    DATASET_DIR, SPLITS, IMG_SIZE = None, ["train", "valid", "test"], 2048
    CLASS_NAMES = {0: "Rotwild", 1: "Rehwild", 2: "Schwarzwild"}

_FNAME_RE = re.compile(r"^(\d+)_(\d+)_jpg")


def parse_flight_frame(filename: str) -> tuple[int, int]:
    """Extract (flight_id, frame_num) from a dataset filename; (-1,-1) if malformed."""
    m = _FNAME_RE.match(Path(filename).name)
    if not m:
        return (-1, -1)
    return (int(m.group(1)), int(m.group(2)))


def yolo_to_box(xc: float, yc: float, w: float, h: float, img_size: int = IMG_SIZE):
    """Normalized YOLO (center,w,h) -> pixel (x0,y0,x1,y1), clipped to the image."""
    x0 = (xc - w / 2) * img_size
    y0 = (yc - h / 2) * img_size
    x1 = (xc + w / 2) * img_size
    y1 = (yc + h / 2) * img_size
    x0 = min(max(x0, 0.0), img_size)
    y0 = min(max(y0, 0.0), img_size)
    x1 = min(max(x1, 0.0), img_size)
    y1 = min(max(y1, 0.0), img_size)
    return (x0, y0, x1, y1)


def long_side_px(w: float, h: float, img_size: int = IMG_SIZE) -> float:
    """Longest box side in pixels from normalized w,h."""
    return float(max(w, h) * img_size)


def load_annotations(dataset_dir: Path | str | None = None,
                     splits=SPLITS) -> pd.DataFrame:
    """Load all YOLO boxes into a tidy DataFrame.

    Columns: split, image_file, label_file, line_num, cls, cls_name,
             xc, yc, w, h, flight_id, frame_num, long_px.
    """
    root = Path(dataset_dir or DATASET_DIR)
    rows = []
    for split in splits:
        ldir = root / split / "labels"
        if not ldir.is_dir():
            continue
        for lp in sorted(ldir.glob("*.txt")):
            image_file = lp.stem + ".jpg"
            fid, fr = parse_flight_frame(image_file)
            text = lp.read_text().strip()
            if not text:
                continue
            for i, line in enumerate(text.split("\n")):
                p = line.split()
                if len(p) < 5:
                    continue
                cls = int(p[0])
                xc, yc, w, h = (float(v) for v in p[1:5])
                rows.append({
                    "split": split, "image_file": image_file, "label_file": lp.name,
                    "line_num": i, "cls": cls, "cls_name": CLASS_NAMES.get(cls, str(cls)),
                    "xc": xc, "yc": yc, "w": w, "h": h,
                    "flight_id": fid, "frame_num": fr,
                    "long_px": long_side_px(w, h),
                })
    return pd.DataFrame(rows)
