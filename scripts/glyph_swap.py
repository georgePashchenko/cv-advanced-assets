"""Replace a glyph in a raster with another glyph copied from the same image.

Usage: glyph_swap.py <in> <out> "sx0,sy0,sx1,sy1:dx0,dy0" [...]
Each spec copies the source box onto the destination box of the same size: the destination is first
filled with a background interpolated from its left/right borders, then the source glyph is blended in
by its contrast against the source background.
"""
import sys

import numpy as np
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
a = np.asarray(Image.open(src).convert("RGB")).astype(float)
orig = a.copy()
for spec in sys.argv[3:]:
    s, d = spec.split(":")
    sx0, sy0, sx1, sy1 = map(int, s.split(","))
    dx0, dy0 = map(int, d.split(","))
    w, h = sx1 - sx0, sy1 - sy0
    glyph = orig[sy0:sy1, sx0:sx1]
    # source background: per-row median of the box border columns
    sbg = np.median(np.concatenate([orig[sy0:sy1, sx0 - 3:sx0], orig[sy0:sy1, sx1:sx1 + 3]], axis=1), axis=1)
    alpha = np.clip(np.abs(glyph - sbg[:, None]).max(axis=2, keepdims=True) / 90, 0, 1)
    # destination background: linear blend between the columns just left and right of the box
    left = np.median(orig[dy0:dy0 + h, dx0 - 3:dx0], axis=1)
    right = np.median(orig[dy0:dy0 + h, dx0 + w:dx0 + w + 3], axis=1)
    t = np.linspace(0, 1, w)[None, :, None]
    dbg = (1 - t) * left[:, None] + t * right[:, None]
    a[dy0:dy0 + h, dx0:dx0 + w] = dbg + alpha * (glyph - sbg[:, None])
Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(out, quality=95)
print("saved", out)
