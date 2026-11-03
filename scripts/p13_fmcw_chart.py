"""Slide 13 chart: FMCW chirps (TX/RX) and range FFT, layout after the Gemini mock-up.

Usage: p13_fmcw_chart.py <out.png>
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1]

BG = "#181a19"
GRID = "#2c302e"
SPINE = "#5a5f5c"
TICK = "#a0a6a3"
FG = "#e8ecea"
TX = "#38bdf8"
RX = "#f97316"
TEAL = "#2dd4bf"

plt.rcParams.update({
    "font.family": "Segoe UI",
    "mathtext.fontset": "dejavusans",
    "font.size": 15,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": FG,
    "xtick.color": TICK,
    "ytick.color": TICK,
})

F0, F1 = 77.0, 78.0  # GHz
TC, PERIOD, TAU = 40.0, 50.0, 8.0  # us; tau exaggerated for legibility
SLOPE = (F1 - F0) / TC  # GHz/us

fig = plt.figure(figsize=(12.64, 8.48), facecolor=BG)
ax = fig.add_axes([0.11, 0.50, 0.87, 0.41], facecolor=BG)
ax2 = fig.add_axes([0.11, 0.115, 0.87, 0.26], facecolor=BG)

for k in range(3):
    t0 = k * PERIOD
    ax.plot([t0, t0 + TC], [F0, F1], color=TX, lw=3, solid_capstyle="round",
            label="TX chirp" if k == 0 else None)
    ax.plot([t0 + TAU, t0 + TAU + TC], [F0, F1], color=RX, lw=3, ls=(0, (3.5, 2)),
            label="RX (затримка 2R/c)" if k == 0 else None)

# tau: horizontal, same frequency, from TX to RX
f_tau = 77.5
t_tx = (f_tau - F0) / SLOPE
ax.annotate("", xy=(t_tx + TAU, f_tau), xytext=(t_tx, f_tau),
            arrowprops=dict(arrowstyle="<|-|>", color=FG, lw=2, mutation_scale=16, shrinkA=0, shrinkB=0))
ax.text(t_tx - 3.0, f_tau, r"$\tau = 2R/c$", color=FG, fontsize=17, ha="right", va="center")

# f_b: vertical, same time, from RX up to TX
t_fb = 34.0
f_txb, f_rxb = F0 + SLOPE * t_fb, F0 + SLOPE * (t_fb - TAU)
ax.annotate("", xy=(t_fb, f_txb), xytext=(t_fb, f_rxb),
            arrowprops=dict(arrowstyle="<|-|>", color=TEAL, lw=2.4, mutation_scale=16, shrinkA=0, shrinkB=0))
ax.text(t_fb + 1.6, (f_txb + f_rxb) / 2, r"$\mathbf{f_b}$", color=TEAL, fontsize=19, va="center")

ax.set_xlim(-3, 152)
ax.set_ylim(76.95, 78.05)
ax.set_xticks(range(0, 141, 20))
ax.set_yticks(np.round(np.arange(77.0, 78.01, 0.2), 1))
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
ax.set_xlabel("час, мкс", labelpad=4)
ax.set_ylabel("частота, ГГц", labelpad=8)
ax.grid(color=GRID, lw=1)
ax.set_axisbelow(True)
ax.text(1.0, -0.24, "τ перебільшено для наочності", transform=ax.transAxes, color="#7d837f",
        fontsize=12, ha="right", va="top")

# Header row: formula on the left, legend on the right
fig.text(0.11, 0.955, r"$\mathbf{f_b = S\cdot\tau}$ — різницева частота", color=TEAL, fontsize=19,
         va="center")
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, fontsize=16,
          labelcolor=FG, handlelength=2.6, columnspacing=1.8, borderaxespad=0)

# Range FFT: S = 25 MHz/us -> f_b = S * 2R / c
S = 25e12  # Hz/s
c = 3e8
fb = lambda r: S * 2 * r / c / 1e6  # MHz
f = np.linspace(0, 25, 2000)
lor = lambda f0, a, w: a / (1 + ((f - f0) / w) ** 2)
spec = lor(fb(30), 0.12, 0.35) + lor(fb(100), 1.0, 0.35) + 0.01
ax2.fill_between(f, 0, spec, color=TEAL, alpha=0.45, lw=0)
ax2.plot(f, spec, color=TEAL, lw=2.4)
for r in (30, 100):
    ax2.text(fb(r), spec[np.argmin(abs(f - fb(r)))] + 0.06, f"{r} м", color=FG, fontsize=17,
             ha="center", va="bottom")
ax2.set_xlim(0, 25)
ax2.set_ylim(0, 1.32)
ax2.set_xticks(range(0, 26, 5))
ax2.set_yticks([])
ax2.set_xlabel(r"$f_b$, МГц → range FFT", labelpad=4)
ax2.grid(axis="x", color=GRID, lw=1)
ax2.set_axisbelow(True)

# Guide: beat frequency of the top plot -> peak in the spectrum
to_fig = fig.transFigure.inverted()
xa, ya = to_fig.transform(ax.transData.transform((t_fb, F0)))
xb, yb = to_fig.transform(ax2.transData.transform((fb(100), 1.27)))
y_mid = 0.405
fig.add_artist(plt.Line2D([xa, xa, xb, xb], [ya, y_mid, y_mid, yb], transform=fig.transFigure,
                          color=TEAL, lw=1.6, ls=(0, (1.5, 2.5))))
ax.plot([t_fb, t_fb], [F0 - 0.05, f_rxb], color=TEAL, lw=1.6, ls=(0, (1.5, 2.5)), clip_on=False)

fig.savefig(out, dpi=200, facecolor=BG)
print("ok", out)
