"""Slide 22 ("Три родини extrinsic-калібрування"): fit three Gemini illustrations into the cards.

Usage: p22_calib_cards.py <src_dir> <out_dir>
Reads ill_target.png, ill_targetless.png, ill_motion.png (Gemini renders without text, prompts in versions/),
crops each to ASPECT around its content, shifts the background to the card fill and fades the borders into it.
"""
import os
import sys

import numpy as np
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
CARD = np.array([30, 33, 36], float)  # card fill, #1e2124
ASPECT = 1.9
CENTER_Y = {"target": 0.50, "targetless": 0.50, "motion": 0.48}  # vertical centre of the content, fraction of height

for name, cy in CENTER_Y.items():
    im = Image.open(os.path.join(src, f"ill_{name}.png")).convert("RGB")
    W, H = im.size
    h = round(W / ASPECT)
    y0 = int(np.clip(cy * H - h / 2, 0, H - h))
    a = np.asarray(im.crop((0, y0, W, y0 + h))).astype(float)
    corners = np.concatenate([a[:20, :20], a[:20, -20:], a[-20:, :20], a[-20:, -20:]]).reshape(-1, 3)
    a += CARD - np.median(corners, axis=0)
    r = 50
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    xx = np.minimum(np.arange(W), np.arange(W)[::-1])[None, :]
    m = np.clip(np.minimum(yy, xx) / r, 0, 1)[..., None]
    a = m * a + (1 - m) * CARD
    Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(os.path.join(out, f"p22_{name}.jpg"), quality=92)
    print(name, W, h)
