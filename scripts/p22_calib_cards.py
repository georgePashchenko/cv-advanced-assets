"""Slide 22 ("Три родини extrinsic-калібрування"): fit three Gemini illustrations into the cards.

Usage: p22_calib_cards.py <src_dir> <out_dir>
Reads ill_target.png, ill_targetless.png, ill_motion.png (Gemini renders without text, prompts in versions/),
crops each to ASPECT around its content box, shifts the background to the card fill and fades the borders into it.
"""
import os
import sys

import numpy as np
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
CARD = np.array([30, 33, 36], float)  # card fill, #1e2124
ASPECT = 1.9
# content box (x0, y0, x1, y1) as fractions of the render; the crop is the smallest ASPECT box around it
CONTENT = {"target": (0.06, 0.08, 0.94, 0.91), "targetless": (0.15, 0.07, 0.85, 0.89),
           "motion": (0.14, 0.11, 0.97, 0.84)}

for name, (fx0, fy0, fx1, fy1) in CONTENT.items():
    im = Image.open(os.path.join(src, f"ill_{name}.png")).convert("RGB")
    W0, H0 = im.size
    cw, ch = (fx1 - fx0) * W0, (fy1 - fy0) * H0
    cw, ch = max(cw, ch * ASPECT), max(ch, cw / ASPECT)
    cx, cy = (fx0 + fx1) / 2 * W0, (fy0 + fy1) / 2 * H0
    x0 = int(np.clip(cx - cw / 2, 0, W0 - cw)); y0 = int(np.clip(cy - ch / 2, 0, H0 - ch))
    a = np.asarray(im.crop((x0, y0, x0 + int(cw), y0 + int(ch)))).astype(float)
    h, W = a.shape[:2]
    corners = np.concatenate([a[:20, :20], a[:20, -20:], a[-20:, :20], a[-20:, -20:]]).reshape(-1, 3)
    a += CARD - np.median(corners, axis=0)
    r = 50
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    xx = np.minimum(np.arange(W), np.arange(W)[::-1])[None, :]
    m = np.clip(np.minimum(yy, xx) / r, 0, 1)[..., None]
    a = m * a + (1 - m) * CARD
    Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(os.path.join(out, f"p22_{name}.jpg"), quality=92)
    print(name, W, h)
