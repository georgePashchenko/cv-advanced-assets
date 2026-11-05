"""Slide 44 (bridge to lesson 12): f·ω·Δt chart for the left card, active-vs-passive illustration for the right card.

Usage: p44_bridge.py <illustration.png> <out_dir>
The illustration is a Gemini render without text (prompts in versions/); this script crops it, matches its
background to the card fill and adds the Ukrainian labels. Both outputs are 3.17:1 to fit the free area of a card.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ill_path, out_dir = sys.argv[1], sys.argv[2]
plt.rcParams["font.family"] = "Segoe UI"

CARD = (30, 33, 36)  # card fill on the slide, #1e2124
CARD_HEX = "#1e2124"
FG = "#e8e8e8"
MUTED = "#8a8d93"
TEAL = "#2ec4b6"
ORANGE = "#f07a1e"
F_LWIR = 320 / np.tan(np.radians(22.5))  # 640 px, HFOV 45° (sim12.py): ≈ 773 px

# ---- left card: shift in px = f · ω · Δt ----
fig, ax = plt.subplots(figsize=(6.33, 2.0), facecolor=CARD_HEX)
fig.subplots_adjust(left=0.09, right=0.84, top=0.85, bottom=0.25)
ax.set_facecolor(CARD_HEX)
dt = np.linspace(0, 40, 200)  # ms
for w, col in [(10, "#7fd8cf"), (30, TEAL), (90, "#1b8f86")]:
    px = F_LWIR * np.radians(w) * dt / 1000
    ax.plot(dt, px, color=col, lw=2)
    # label at the right edge, or where the line leaves the top
    if px[-1] <= 20:
        ax.text(40.8, px[-1], f"{w} °/с", color=col, fontsize=11, va="center")
    else:
        x_top = 20 / (F_LWIR * np.radians(w) / 1000)
        ax.text(x_top + 0.6, 19.2, f"{w} °/с", color=col, fontsize=11, va="top")
ax.axhline(1, color=ORANGE, lw=1.2, ls="--")
ax.text(40.8, 1, "1 px", color=ORANGE, fontsize=11, va="center")
p = F_LWIR * np.radians(30) * 0.010
ax.plot([10], [p], "o", color="white", ms=6, zorder=5)
ax.annotate(f"30 °/с × 10 мс ≈ {p:.0f} px", (10, p), (14.5, 10.5), color=FG, fontsize=11,
            arrowprops=dict(arrowstyle="-", color=FG, lw=1))
ax.set_xlim(0, 40)
ax.set_ylim(0, 20)
ax.set_xlabel("розсинхрон Δt, мс", color=MUTED, fontsize=11, labelpad=2)
ax.set_ylabel("зсув, px", color=MUTED, fontsize=11)
ax.set_title(f"зсув = f · ω · Δt,   f ≈ {F_LWIR:.0f} px (LWIR 640 px, HFOV 45°)",
             color=FG, fontsize=11.5, loc="left", pad=8)
ax.tick_params(colors=MUTED, labelsize=10)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
fig.savefig(os.path.join(out_dir, "p44_fwdt.png"), dpi=330, facecolor=CARD_HEX)
plt.close(fig)

# ---- right card: illustration with labels ----
im = Image.open(ill_path).convert("RGB")
W, H = im.size
im = im.crop((0, round(0.186 * H), W, round(0.186 * H) + round(W / 3.17)))
a = np.asarray(im).astype(float)
bg = np.median(np.concatenate([a[:20, :20].reshape(-1, 3), a[:20, -20:].reshape(-1, 3)]), axis=0)
a += np.array(CARD) - bg  # match the Gemini background to the card fill
# fade the borders to the exact card colour so no rectangle edge shows on the slide
h, w = a.shape[:2]
r = 60
yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
xx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
m = np.clip(np.minimum(yy, xx) / r, 0, 1)[..., None]
a = m * a + (1 - m) * np.array(CARD)
im = Image.fromarray(a.clip(0, 255).astype(np.uint8))

d = ImageDraw.Draw(im)
font = ImageFont.truetype(r"C:\Windows\Fonts\segoeui.ttf", 92)
bold = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", 92)
y = im.size[1] - 45
for cx, col, head, rest in [(w * 0.25, TEAL, "Активні: ", "світять і ловлять відбиття"),
                            (w * 0.75, ORANGE, "Пасивний: ", "лише приймає тепло")]:
    tw = d.textlength(head, font=bold) + d.textlength(rest, font=font)
    x = cx - tw / 2
    d.text((x, y), head, font=bold, fill=col, anchor="ls")
    d.text((x + d.textlength(head, font=bold), y), rest, font=font, fill=(232, 232, 232), anchor="ls")
im.save(os.path.join(out_dir, "p44_active_passive.jpg"), quality=93)
print("ok", im.size)
