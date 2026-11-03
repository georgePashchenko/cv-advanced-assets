"""Re-lay out the slide-10 infographic: title, 5 cards full width, datasheet as a bottom strip.

Usage: relayout.py <in v6 6336x2688> <out> <preview>
All text comes from crops of the verified image; nothing is re-rendered.
"""
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

src, out, preview = sys.argv[1:4]
im = Image.open(src).convert("RGB")
W = im.width
BG = (19, 18, 21)
CARD_BG = np.array([19, 29, 21], float)
GREEN = (88, 222, 120)
M = 60  # outer margin
S = 1.194  # scale shared by cards and strip pieces, keeps font sizes equal


def feather(img, r=30):
    """Alpha mask fading the crop edges into the background."""
    w, h = img.size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).rectangle((r, r, w - r, h - r), fill=255)
    return m.filter(ImageFilter.GaussianBlur(r / 2))


def scaled(box, s=S):
    c = im.crop(box)
    return c.resize((round(c.width * s), round(c.height * s)), Image.LANCZOS)


def text_layer(box, s=S):
    """Crop of card text keyed against the card background: (RGB, alpha)."""
    c = np.asarray(scaled(box, s)).astype(float)
    diff = np.abs(c - CARD_BG).max(axis=2)
    alpha = np.clip((diff - 32) / 50, 0, 1)
    return Image.fromarray(c.astype(np.uint8)), Image.fromarray((alpha * 255).astype(np.uint8))


title = im.crop((200, 90, 6136, 460))
cards = scaled((48, 482, 5250, 2352), (W - 2 * M) / (5250 - 48))

# Datasheet pieces: (x0, y0, x1, y1) in the source, from find_blocks.py
P_TITLE = (5318, 548, 6100, 740)
P_WARN = (5296, 848, 6125, 1122)
P_HEAD = (5296, 1192, 6118, 1290)
P_ITEMS = [(5314, 1357, 5976, 1535), (5313, 1604, 6026, 1782),
           (5312, 1845, 6098, 2028), (5314, 2096, 6166, 2276)]

pad = 60
t_rgb, t_a = text_layer(P_TITLE)
w_rgb, w_a = text_layer(P_WARN)
h_rgb, h_a = text_layer(P_HEAD)
items = [text_layer(b) for b in P_ITEMS]
left_w = max(t_rgb.width, w_rgb.width)
strip_h = pad + t_rgb.height + 20 + w_rgb.height + pad

y_title = 0
y_cards = y_title + title.height - 10
y_strip = y_cards + cards.height + 30
H = y_strip + strip_h + 50

canvas = Image.new("RGB", (W, H), BG)
canvas.paste(title, ((W - title.width) // 2, y_title), feather(title))
canvas.paste(cards, (M, y_cards), feather(cards, 24))

# Strip frame with neon glow
sx0, sy0, sx1, sy1 = M + 6, y_strip, W - M - 6, y_strip + strip_h
glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
gd.rounded_rectangle((sx0, sy0, sx1, sy1), radius=40, outline=GREEN + (255,), width=34)
glow = glow.filter(ImageFilter.GaussianBlur(30))
canvas.paste(glow, (0, 0), glow)
canvas.paste(glow, (0, 0), glow)
d = ImageDraw.Draw(canvas)
d.rounded_rectangle((sx0, sy0, sx1, sy1), radius=40, fill=tuple(CARD_BG.astype(int)), outline=GREEN, width=7)

# Left column: title + warning
lx = sx0 + pad
canvas.paste(t_rgb, (lx, sy0 + pad), t_a)
canvas.paste(w_rgb, (lx, sy0 + pad + t_rgb.height + 20), w_a)

# Divider
dx = lx + left_w + 70
d.line((dx, sy0 + 50, dx, sy1 - 50), fill=(60, 140, 80), width=4)

# Right area: header + four item columns, centered vertically
rx0, rx1 = dx + 80, sx1 - pad
item_h = max(r.height for r, _ in items)
block_h = h_rgb.height + 40 + item_h
by = sy0 + (strip_h - block_h) // 2
canvas.paste(h_rgb, (rx0, by), h_a)
col_w = (rx1 - rx0) // 4
for i, (rgb, a) in enumerate(items):
    canvas.paste(rgb, (rx0 + i * col_w, by + h_rgb.height + 40), a)

canvas.save(out, quality=95)
canvas.resize((1584, round(1584 * H / W)), Image.LANCZOS).save(preview)
print(canvas.size, f"aspect {W / H:.3f}")
