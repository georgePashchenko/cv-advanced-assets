"""Slide 24: covariance propagation formulas rendered as math, colour-coded like the chart on the slide.

Usage: p24_formula.py <out.png>
Transparent PNG for the dark card. Numbers: f = 721.5 px, σθ = 0.1°, σt = 2 cm (Demo 1).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
plt.rcParams.update({"font.family": "Segoe UI", "mathtext.fontset": "dejavusans"})
FG, MUTED = "#e6e8eb", "#9aa0a6"
ROT, TR, SUM = "#38bdf8", "#f07a1e", "#2ec4b6"  # chart colours: rotation, translation, total
F, S_TH, S_T = 721.5, np.radians(0.1), 0.02

fig = plt.figure(figsize=(7.6, 2.1))
fig.patch.set_alpha(0)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.canvas.draw()
r = fig.canvas.get_renderer()


def row(x, y, parts, size, gap=0.004):
    """Draw [(text, colour), ...] left to right; return the x extents of each part."""
    spans = []
    for s, c in parts:
        t = ax.text(x, y, s, color=c, fontsize=size, va="baseline", ha="left")
        bb = t.get_window_extent(r).transformed(ax.transData.inverted())
        spans.append((bb.x0, bb.x1))
        x = bb.x1 + gap
    return spans


# general form
row(0.0, 0.76, [(r"$\Sigma_{uv} = J\,\Sigma_\xi\,J^{\mathsf{T}}$", FG),
                (r"$J = \partial(u,v)\,/\,\partial\xi$", MUTED)], 21, gap=0.045)
ax.text(0.0, 0.58, "6 параметрів калібрування  ξ = (δθx, δθy, δθz, δtx, δty, δtz)  →  еліпс 2×2 на кадрі",
        color=MUTED, fontsize=11.5, va="baseline")

# on-axis approximation with colour-coded terms
sp = row(0.0, 0.28, [(r"$\sigma_u^2 \approx$", FG), (r"$(f\,\sigma_\theta)^2$", ROT), (r"$+$", FG),
                     (r"$(f\,\sigma_t\,/\,Z)^2$", TR)], 21)
rot_px = F * S_TH
ax.text((sp[1][0] + sp[1][1]) / 2, 0.08, f"поворот: {rot_px:.2f} px\nна будь-якій Z", color=ROT, fontsize=11,
        ha="center", va="baseline", linespacing=1.15)
ax.text((sp[3][0] + sp[3][1]) / 2, 0.08, f"зсув: {F * S_T / 5:.1f} px на 5 м,\n{F * S_T / 30:.2f} px на 30 м",
        color=TR, fontsize=11, ha="center", va="baseline", linespacing=1.15)
x_eq = sp[3][1] + 0.05
for k, Z in enumerate((5, 30)):
    s = np.hypot(rot_px, F * S_T / Z)
    ax.text(x_eq, 0.33 - k * 0.2, rf"$\sigma_u$ = {s:.2f} px  @ {Z} м", color=SUM, fontsize=15, va="baseline",
            fontweight="bold")
fig.savefig(out, dpi=250, transparent=True, bbox_inches="tight", pad_inches=0.05)
print("saved", out)
