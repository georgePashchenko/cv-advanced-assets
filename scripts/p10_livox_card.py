"""Slide 10 infographic, card 5: replace the rosette (Mid-40/Avia pattern) with a Mid-360-style coverage view.

Usage: p10_livox_card.py <in> <out>
Mid-360 is a 360 x 59 deg (-7..52) panoramic sensor, so its coverage is shown as two azimuth x elevation
strips: one 10 Hz frame (100 ms) and 1 s of accumulation. The pattern itself is illustrative
(incommensurate sweeps, as in p10_scan_patterns.py but with a smaller azimuth wobble), at the datasheet rate of 200k points/s.
"""
import io
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

plt.rcParams["font.family"] = "Bahnschrift"

src, out = sys.argv[1], sys.argv[2]
X0, Y0, X1, Y1 = 5090, 1085, 6222, 2505  # card 5 area below the "Конструкція" text
FONT = (r"C:\Windows\Fonts\bahnschrift.ttf", "SemiLight SemiCondensed")
WHITE = (240, 240, 245)
MAGENTA = "#e070ff"
RATE = 200.0  # points per ms (200k pts/s)


def livox(duration_ms):
    t = np.arange(0, duration_ms, 1 / RATE)
    s = t / 1000.0
    az = (s * 360 * 10 + 12 * np.sin(2 * np.pi * 37.3 * s)) % 360
    el = 22.5 + 29.5 * np.sin(2 * np.pi * 241.8034 * s + 2.1 * np.sin(2 * np.pi * 7.07 * s))
    return az, el


def strip(duration_ms, w, h):
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0.12, 0.22, 0.83, 0.71])
    ax.set_facecolor("#0d0612")
    az, el = livox(duration_ms)
    ax.scatter(az, el, s=0.25 if duration_ms > 200 else 0.9, c=MAGENTA, alpha=0.4 if duration_ms > 200 else 1,
               linewidths=0, rasterized=True)
    ax.set_xlim(0, 360)
    ax.set_ylim(-7, 52)
    ax.set_xticks([0, 180, 360])
    ax.set_xticklabels(["0°", "180°", "360°"])
    ax.set_yticks([-7, 52])
    ax.set_yticklabels(["-7°", "52°"])
    ax.tick_params(colors="#c9c2d0", labelsize=42, length=4, pad=3)
    ax.tick_params(axis="y", pad=10)
    for sp in ax.spines.values():
        sp.set_color("#e6e0ea")
        sp.set_linewidth(1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, transparent=True)
    plt.close(fig)
    return Image.open(buf).convert("RGBA")


im = Image.open(src).convert("RGB")
a = np.asarray(im).astype(float)
# card background: blend per-column medians of the strips just above and below the area
top = np.median(a[Y0 - 10:Y0 - 2, X0:X1], axis=0)
bot = np.median(a[Y1 + 2:Y1 + 10, X0:X1], axis=0)
w = np.linspace(0, 1, Y1 - Y0)[:, None, None]
a[Y0:Y1, X0:X1] = (1 - w) * top[None] + w * bot[None]
im = Image.fromarray(a.clip(0, 255).astype(np.uint8)).convert("RGBA")

W = X1 - X0
font = ImageFont.truetype(FONT[0], 88)
font.set_variation_by_name(FONT[1])
small = ImageFont.truetype(FONT[0], 68)
small.set_variation_by_name(FONT[1])
d = ImageDraw.Draw(im)

y = Y0 + 40
for dur, label in [(100, "За 100 мс (1 кадр)"), (1000, "За 1 с (10 кадрів)")]:
    d.text((X0 + W // 2, y), label, font=font, fill=WHITE, anchor="mt")
    img = strip(dur, W - 20, 440)
    im.alpha_composite(img, (X0 + 10, y + 105))
    y += 105 + 440 + 30

d.text((X0 + W // 2, y + 10), "Патерн не повторюється:\nпокриття FoV росте з часом",
       font=small, fill=WHITE, anchor="ma", align="center", spacing=12)
im.convert("RGB").save(out, quality=93)
print("saved", out)
