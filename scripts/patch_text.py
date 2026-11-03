"""Erase a text box and redraw lines over a background blended from clean strips above/below.

Usage: patch_text.py <in> <out> <preview> x0 y0 x1 y1 tx ty size step "line1|line2|..."
(ty is the baseline of the first line)
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, out, preview = sys.argv[1:4]
x0, y0, x1, y1, tx, ty, size, step = map(int, sys.argv[4:12])
lines = sys.argv[12].split("|")

a = np.asarray(Image.open(src).convert("RGB")).astype(float)


def strip(ya, yb):
    s = np.median(a[ya:yb, x0:x1], axis=0)
    k = np.ones(41) / 41
    return np.stack([np.convolve(np.pad(s[:, c], 20, mode="edge"), k, mode="valid") for c in range(3)], 1)


top, bot = strip(y0 - 14, y0 - 2), strip(y1 + 4, y1 + 16)
w = np.linspace(0, 1, y1 - y0)[:, None, None]
a[y0:y1, x0:x1] = (1 - w) * top[None] + w * bot[None]
im = Image.fromarray(a.clip(0, 255).astype(np.uint8))

d = ImageDraw.Draw(im)
font = ImageFont.truetype(r"C:\Windows\Fonts\bahnschrift.ttf", size)
font.set_variation_by_name("SemiLight SemiCondensed")
for i, line in enumerate(lines):
    d.text((tx, ty + i * step), line, font=font, fill=(232, 232, 232), anchor="ls")  # ty = baseline

im.save(out, quality=95)
im.crop((max(0, x0 - 150), max(0, y0 - 250), min(im.width, x1 + 150), min(im.height, y1 + 250))).save(preview)
