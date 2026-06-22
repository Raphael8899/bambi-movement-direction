"""Keyboard annotation tool for the thermal animal crops.

Run it from the folder that holds manifest.csv and the crops:

    python annotate.py                       # uses ./manifest.csv and ./labels.csv
    python annotate.py manifest.csv out.csv andreas

Keys
    direction:
        5 = rotate the body/blur line to the angle (press repeatedly; 8 steps of 22.5 deg)
        1 = arrow at one end     2 = arrow at the other end     3 = no arrow (axis only)
        0 = nothing usable in this crop
    motion:   s stationary   d moving   u unsure
    Enter/Space next   Backspace previous   Esc save & quit

Labels are saved after every crop, and reopening resumes at the first unlabelled crop.
"""
import math
import os
import sys
import time
import tkinter as tk
from tkinter import font as tkfont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import label_store as ls
from PIL import Image, ImageTk, ImageDraw

DISPLAY = 500
THUMB_MAX = 160

MOTION_KEYS = {"s": "stationary", "d": "moving", "u": "unsure"}
# direction keys map to (head state); 5 rotates the orientation, 0 = nothing usable
HEAD_KEYS = {"1": "rev", "2": "fwd", "3": "none"}

BG, PANEL, FG, MUTED = "#1e1e1e", "#262626", "#e6e6e6", "#9a9a9a"
ACCENT, GOOD, WARN, ARROW = "#4fc3f7", "#7bd88f", "#ffb454", "#ff5252"


class App:
    def __init__(self, root, store):
        self.root = root
        self.store = store
        self.t0 = time.time()
        self._photo = None
        self._thumb = None

        root.configure(bg=BG)
        root.minsize(900, 640)
        self._fonts()
        self._layout()
        self._bind()
        self.refresh()
        self._tick()

    def _fonts(self):
        self.f_big = tkfont.Font(family="Segoe UI", size=14, weight="bold")
        self.f_med = tkfont.Font(family="Segoe UI", size=12)
        self.f_small = tkfont.Font(family="Segoe UI", size=10)
        self.f_mono = tkfont.Font(family="Consolas", size=11)

    def _layout(self):
        left = tk.Frame(self.root, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        right = tk.Frame(self.root, bg=PANEL, width=340)
        right.pack(side="right", fill="y", padx=(0, 12), pady=12)
        right.pack_propagate(False)

        self.canvas = tk.Canvas(left, width=DISPLAY, height=DISPLAY, bg="#000",
                                highlightthickness=1, highlightbackground="#444")
        self.canvas.pack(pady=(0, 8))

        row = tk.Frame(left, bg=BG)
        row.pack(anchor="w")
        tk.Label(row, text="real size:", bg=BG, fg=MUTED, font=self.f_small).pack(side="left", padx=(0, 8))
        self.thumb = tk.Label(row, bg=BG, bd=1, relief="solid")
        self.thumb.pack(side="left")

        self.status = tk.Label(left, text="", bg=BG, fg=FG, font=self.f_big, anchor="w")
        self.status.pack(anchor="w", pady=(12, 0))
        self.missing = tk.Label(left, text="", bg=BG, fg=WARN, font=self.f_med, anchor="w")
        self.missing.pack(anchor="w", pady=(2, 0))

        self.progress = tk.Label(right, text="", bg=PANEL, fg=ACCENT, font=self.f_big, justify="left")
        self.progress.pack(anchor="w", padx=14, pady=(14, 2))
        self.elapsed = tk.Label(right, text="", bg=PANEL, fg=MUTED, font=self.f_med)
        self.elapsed.pack(anchor="w", padx=14, pady=(0, 10))
        self.selection = tk.Label(right, text="", bg=PANEL, fg=FG, font=self.f_med, justify="left")
        self.selection.pack(anchor="w", padx=14, pady=(0, 12))

        tk.Label(right, text="DIRECTION", bg=PANEL, fg=MUTED, font=self.f_small).pack(anchor="w", padx=14)
        tk.Label(right, text=("5  rotate the line to the angle\n"
                              "1  arrow at one end\n"
                              "2  arrow at the other end\n"
                              "3  no arrow (axis only)\n"
                              "0  nothing usable"),
                 bg=PANEL, fg=FG, font=self.f_mono, justify="left").pack(anchor="w", padx=24, pady=(2, 0))

        tk.Label(right, text="MOTION", bg=PANEL, fg=MUTED, font=self.f_small).pack(anchor="w", padx=14, pady=(12, 0))
        tk.Label(right, text="s stationary   d moving   u unsure", bg=PANEL, fg=FG,
                 font=self.f_mono, justify="left").pack(anchor="w", padx=24)
        tk.Label(right, text="NAVIGATION", bg=PANEL, fg=MUTED, font=self.f_small).pack(anchor="w", padx=14, pady=(12, 0))
        tk.Label(right, text="Enter/Space next\nBackspace previous\nEsc save & quit",
                 bg=PANEL, fg=FG, font=self.f_mono, justify="left").pack(anchor="w", padx=24)

    def _bind(self):
        for d in "0123456789":
            self.root.bind(d, self._digit)
            self.root.bind(f"<KP_{d}>", self._digit)
        for k in MOTION_KEYS:
            self.root.bind(k, self._motion)
            self.root.bind(k.upper(), self._motion)
        for seq in ("<Return>", "<KP_Enter>", "<space>"):
            self.root.bind(seq, self._next)
        self.root.bind("<BackSpace>", lambda e: (self.store.back(), self.refresh()))
        self.root.bind("<Escape>", self._quit)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _digit_of(self, event):
        ks = event.keysym
        if ks.startswith("KP_") and ks[3:].isdigit():
            return ks[3:]
        if ks.isdigit():
            return ks
        return event.char if (event.char and event.char.isdigit()) else None

    def _digit(self, event):
        d = self._digit_of(event)
        cur = self.store.label()["direction_class"]
        if d == "5":                                  # rotate the orientation (angle)
            self.store.set_direction(ls.rotate_orientation(cur))
        elif d in HEAD_KEYS:                           # 1/2 = arrow at an end, 3 = no arrow
            self.store.set_direction(ls.set_head(cur, HEAD_KEYS[d]))
        elif d == "0":                                 # nothing usable
            self.store.set_none()
        else:
            return
        self._changed()

    def _motion(self, event):
        state = MOTION_KEYS.get(event.keysym.lower())
        if state:
            self.store.set_motion(state)
            self._changed()

    def _changed(self):
        try:
            self.store.save()
        except Exception as e:
            self.status.config(text=f"save failed: {e}", fg=WARN)
        self.refresh()

    def _next(self, event=None):
        if not self.store.is_complete():
            self.status.config(text="still missing: " + " + ".join(self.store.missing()), fg=WARN)
            return
        before = self.store.index
        self.store.advance()
        if self.store.index == before:
            self.status.config(text="last crop reached — Esc to save & quit", fg=GOOD)
        self.refresh()

    def _quit(self, event=None):
        try:
            self.store.save()
        except Exception:
            pass
        self.root.destroy()

    def refresh(self):
        idx, n = self.store.index, self.store.count()
        cid, sp = self.store.crop_id(), self.store.class_id()
        self.root.title(f"annotate  |  {cid}  |  class {sp}")

        img = self._load(self.store.image_path())
        self._draw_main(img)
        self._draw_thumb(img)

        self.progress.config(text=f"{idx + 1} / {n}\ncompleted: {self.store.completed_count()}")
        self._update_elapsed()

        lab = self.store.label()
        motion = lab["motion_state"] or "—"
        dcls = lab["direction_class"]
        dname = ls.COMPASS_NAMES.get(dcls, "—") if dcls is not None else "—"
        self.selection.config(text=f"crop: {cid}\nclass: {sp}\n\nmotion:    {motion}\ndirection: {dname}")

        if self.store.is_complete():
            self.status.config(text="complete — Enter/Space for next", fg=GOOD)
            self.missing.config(text="")
        else:
            self.status.config(text="incomplete", fg=WARN)
            self.missing.config(text="missing: " + " + ".join(self.store.missing()))

    def _load(self, path):
        try:
            img = Image.open(path)
            img.load()
            return img.convert("RGB")
        except Exception:
            ph = Image.new("RGB", (120, 120), (60, 20, 20))
            d = ImageDraw.Draw(ph)
            d.line((0, 0, 119, 119), fill=(255, 80, 80), width=3)
            d.line((0, 119, 119, 0), fill=(255, 80, 80), width=3)
            return ph

    def _draw_main(self, img):
        big = img.resize((DISPLAY, DISPLAY), Image.NEAREST)
        dcls = self.store.label()["direction_class"]
        i = ls.axis_index(dcls)
        if i is not None:                                   # new-style direction
            big = big.copy()
            if ls.head_of(dcls) == "none":
                self._axis_line(big, ls.AXIS_STEP * i)      # double-headed line, no arrow
            else:
                self._arrow(big, ls.LabelStore.direction_class_to_deg(dcls))   # arrow at an end
        elif isinstance(dcls, int) and 0 <= dcls <= 7:      # legacy compass heading
            big = big.copy()
            self._arrow(big, ls.LabelStore.direction_class_to_deg(dcls))
        elif dcls == ls.DIR_AXIS_ONLY:                       # legacy generic axis-only
            big = big.copy()
            self._axis_line(big, 0.0)
        # None / DIR_NONE -> no overlay
        self._photo = ImageTk.PhotoImage(big)
        self.canvas.delete("all")
        self.canvas.create_image(DISPLAY // 2, DISPLAY // 2, image=self._photo)

    def _axis_line(self, img, deg):
        """Double-headed line (no arrowhead) marking an axis whose head is unknown."""
        rad = math.radians(deg)
        dx, dy = math.sin(rad), -math.cos(rad)   # 0 deg = vertical (up), increasing clockwise
        c = DISPLAY / 2
        L = DISPLAY * 0.34
        d = ImageDraw.Draw(img)
        d.line([(c - dx * L, c - dy * L), (c + dx * L, c + dy * L)], fill=WARN, width=6)
        d.ellipse([c - 9, c - 9, c + 9, c + 9], outline=WARN, width=4)

    def _arrow(self, img, deg):
        rad = math.radians(deg)
        dx, dy = math.sin(rad), -math.cos(rad)   # 0 deg points up, clockwise
        c = DISPLAY / 2
        L = DISPLAY * 0.32
        tip = (c + dx * L, c + dy * L)
        tail = (c - dx * L, c - dy * L)
        d = ImageDraw.Draw(img)
        d.line([tail, tip], fill=ARROW, width=6)
        head = DISPLAY * 0.10
        for s in (+1, -1):
            a = math.atan2(dy, dx) + s * math.radians(150)
            d.line([tip, (tip[0] + math.cos(a) * head, tip[1] + math.sin(a) * head)], fill=ARROW, width=6)

    def _draw_thumb(self, img):
        w, h = img.size
        scale = min(1.0, THUMB_MAX / max(w, h))
        thumb = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.NEAREST)
        self._thumb = ImageTk.PhotoImage(thumb)
        self.thumb.config(image=self._thumb)

    def _update_elapsed(self):
        e = int(time.time() - self.t0)
        m, s = divmod(e, 60)
        h, m = divmod(m, 60)
        done = self.store.completed_count()
        rate = done / (e / 60.0) if e > 0 and done else 0.0
        self.elapsed.config(text=f"elapsed {h:d}:{m:02d}:{s:02d}   ({rate:.1f}/min)")

    def _tick(self):
        self._update_elapsed()
        self.root.after(1000, self._tick)


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))
    manifest = argv[1] if len(argv) > 1 else os.path.join(here, "manifest.csv")
    labels = argv[2] if len(argv) > 2 else os.path.join(here, "labels.csv")
    annotator = argv[3] if len(argv) > 3 else "andreas"

    if not os.path.exists(manifest):
        print(f"manifest not found: {manifest}", file=sys.stderr)
        return 2
    store = ls.LabelStore(manifest, labels, annotator=annotator)
    if store.count() == 0:
        print("nothing to annotate", file=sys.stderr)
        return 1
    root = tk.Tk()
    App(root, store)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
