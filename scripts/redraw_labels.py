"""Redraw short labels on a raster: background from strips above/below each box, then new text.

Usage: redraw_labels.py <in> <out>   (label specs are in LABELS below; coordinates of radar_v2.jpg, slide 4)
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
THIN = (r"C:\Windows\Fonts\bahnschrift.ttf", "Light SemiCondensed")
MONO = (r"C:\Windows\Fonts\consola.ttf", None)
# (box x0, y0, x1, y1, text x, baseline y, font, size, colour, text)
LABELS = [
    (834, 466, 896, 498, 839, 491, THIN, 27, (232, 236, 236), "300°"),
    (1050, 253, 1110, 285, 1055, 279, THIN, 27, (232, 236, 236), "330°"),
    (2276, 244, 2372, 274, 2279, 268, MONO, 26, (205, 70, 70), "(HIGH)"),
]

a = np.asarray(Image.open(src).convert("RGB")).astype(float)
for x0, y0, x1, y1, *_ in LABELS:
    top = np.median(a[y0 - 6:y0 - 1, x0:x1], axis=0)
    bot = np.median(a[y1 + 1:y1 + 6, x0:x1], axis=0)
    w = np.linspace(0, 1, y1 - y0)[:, None, None]
    a[y0:y1, x0:x1] = (1 - w) * top[None] + w * bot[None]
im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
d = ImageDraw.Draw(im)
for *_, tx, ty, (path, var), size, col, text in LABELS:
    font = ImageFont.truetype(path, size)
    if var:
        font.set_variation_by_name(var)
    d.text((tx, ty), text, font=font, fill=col, anchor="ls")
im.save(out, quality=95)
print("saved", out)
