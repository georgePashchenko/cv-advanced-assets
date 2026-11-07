"""Slide 24: covariance propagation formulas rendered as math, colour-coded like the chart on the slide.

Usage: p24_formula.py <out.png>
Transparent PNG for the dark card. Numbers: f = 721.5 px, σθ = 0.1°, σt = 2 cm (Demo 1); the totals per
distance are in the table next to it on the slide.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]
plt.rcParams.update({"font.family": "Segoe UI", "mathtext.fontset": "dejavusans"})
FG, MUTED = "#e6e8eb", "#9aa0a6"
ROT, TR = "#38bdf8", "#f07a1e"  # chart colours: rotation, translation
F, S_TH, S_T = 721.5, np.radians(0.1), 0.02

fig = plt.figure(figsize=(7.6, 2.5))
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
row(0.0, 0.74, [(r"$\Sigma_{uv} = J\,\Sigma_\xi\,J^{\mathsf{T}}$", FG),
                (r"$J = \partial(u,v)\,/\,\partial\xi$", MUTED)], 24, gap=0.05)
ax.text(0.0, 0.53, "ξ = (δθx, δθy, δθz, δtx, δty, δtz): 6 параметрів → еліпс 2×2 на кадрі",
        color=MUTED, fontsize=14, va="baseline")

# on-axis approximation with colour-coded terms, the value of each term to the right
sp = row(0.0, 0.17, [(r"$\sigma_u^2 \approx$", FG), (r"$(f\,\sigma_\theta)^2$", ROT), (r"$+$", FG),
                     (r"$(f\,\sigma_t\,/\,Z)^2$", TR)], 24)
x = sp[-1][1] + 0.04
ax.text(x, 0.25, f"поворот: {F * S_TH:.2f} px на будь-якій Z", color=ROT, fontsize=14, va="baseline")
ax.text(x, 0.09, f"зсув: {F * S_T / 5:.1f} px на 5 м, {F * S_T / 30:.2f} px на 30 м", color=TR, fontsize=14,
        va="baseline")
fig.savefig(out, dpi=250, transparent=True, bbox_inches="tight", pad_inches=0.05)
print("saved", out)
