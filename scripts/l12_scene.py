"""Lesson 12: slide images from one Gemini scene (road, walker ~8 m, van, house, poles, metal sheet).

Usage: l12_scene.py <versions_dir> <out_dir> <slide> [...]
The renders in versions/ share one geometry: base_rgb (overcast day) and Gemini edits of it, i.e. rgb_night,
rgb_smoke (v2), t_night, t_cross, t_smoke (LWIR, white-hot). Prompts are next to them. Effects that carry
numbers on the slides (blur, AGC, FPN, edge shifts) are applied here, not by Gemini.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
FONT = "C:/Windows/Fonts/seguisb.ttf"
CARD = (30, 33, 36)
rng = np.random.default_rng(12)


def load(name, gray=False):
    im = Image.open(os.path.join(src, name + ".png"))
    return np.asarray(im.convert("L" if gray else "RGB")).astype(float) / 255


def night(rgb):
    """Visible camera at 23:00 without lighting: little signal, high gain -> dark and grainy."""
    g = 0.22 * rgb ** 1.6 * np.array([0.85, 1.0, 0.9])
    g += rng.normal(0, 0.035, g.shape[:2])[..., None] + rng.normal(0, 0.015, g.shape)
    return g.clip(0, 1)


def stretch(t, lo=1, hi=99.7, gamma=1.4):
    a, b = np.percentile(t, [lo, hi])
    return np.clip((t - a) / (b - a), 0, 1) ** gamma


def palette(t, cmap="inferno"):
    return plt.get_cmap(cmap)(stretch(t))[..., :3]


def tag(im, xy, text, size):
    d = ImageDraw.Draw(im, "RGBA")
    f = ImageFont.truetype(FONT, size)
    x0, y0, x1, y1 = d.textbbox(xy, text, font=f)
    p = size // 3
    d.rounded_rectangle((x0 - p, y0 - p, x1 + p, y1 + p), radius=p, fill=(0, 0, 0, 150))
    d.text(xy, text, font=f, fill=(232, 236, 234))


def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def split(left, right, box, cut):
    """Crop box (x0, y0, x1, y1) of two aligned renders, left part from `left`, right part from `right`."""
    x0, y0, x1, y1 = box
    a = np.concatenate([left[y0:y1, x0:cut], right[y0:y1, cut:x1]], axis=1)
    im = to_img(a)
    d = ImageDraw.Draw(im)
    d.line([(cut - x0, 0), (cut - x0, y1 - y0)], fill=(230, 232, 235), width=4)
    s = (y1 - y0) // 13
    tag(im, (s, s), "RGB", s)
    tag(im, (cut - x0 + s, s), "LWIR", s)
    return im


def p4():
    base, smoke = load("base_rgb_v1"), load("rgb_smoke_v2")
    tn, tc, ts = (load(n, gray=True) for n in ("t_night_v1", "t_cross_v1", "t_smoke_v1"))
    box, cut = (150, 330, 2290, 1320), 1150  # aspect 2.16 = card image; van left of the cut, walker right
    for name, l, r in (("night", night(base), palette(tn)), ("cross", base, palette(tc)),
                       ("smoke", smoke, palette(ts))):
        split(l, r, box, cut).save(os.path.join(out, f"p4_{name}.jpg"), quality=92)


for s in sys.argv[3:]:
    globals()[s]()
    print("done", s)
