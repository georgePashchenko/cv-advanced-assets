"""Slide 13 ("Чому FMCW, а не імпульс — і що таке Doppler"): two transparent PNGs.

Usage: p13_fmcw_why.py <out_dir>
p13_waveforms.png — pulsed vs FMCW transmit signal on one time axis (schematic, not to scale).
p13_doppler_vr.png — velocity v split into v_r along the beam and v_perp; Doppler sees only v_r = v·cos θ.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

out_dir = sys.argv[1]
plt.rcParams["font.family"] = "Segoe UI"

FG = "#e6e8eb"
BODY = "#c3c7cd"
MUTED = "#8a8d93"
CYAN = "#38bdf8"
PULSE = "#f2a12a"
RED = "#ef6b6b"


# ---- waveforms: pulsed (high peak, then silence) vs FMCW (low, continuous, rising frequency) ----
fig, axes = plt.subplots(2, 1, figsize=(9.6, 1.8), sharex=True)
fig.patch.set_alpha(0)
fig.subplots_adjust(left=0.19, right=0.65, top=0.97, bottom=0.03, hspace=0.3)
t = np.linspace(0, 3, 6000)

ax = axes[0]
y = np.zeros_like(t)
for t0 in (0.05, 1.55):
    m = (t >= t0) & (t < t0 + 0.12)
    y[m] = np.sin(2 * np.pi * 60 * (t[m] - t0))
ax.plot(t, y, color=PULSE, lw=1.1)
ax.axvspan(0.05, 0.17, color=RED, alpha=0.25, lw=0)
ax.set_ylim(-1.2, 1.2)
ax.text(-0.06, 0, "Імпульсний", color=PULSE, fontsize=17, fontweight="bold", ha="right", va="center",
        transform=ax.get_yaxis_transform())
ax.text(0.85, 0.45, "тиша: слухає ехо", color=MUTED, fontsize=15, ha="center", va="center")
ax.text(3.08, 0, "пік: сотні Вт – кВт\nпоки передає — не чує", color=BODY, fontsize=15, va="center")

ax = axes[1]
tc = 0.5  # chirp period, schematic
tau = t % tc
y = 0.32 * np.sin(2 * np.pi * (8 * tau + 40 * tau ** 2 / tc) / tc * tc)
ax.plot(t, y, color=CYAN, lw=1.0)
ax.set_ylim(-1.2, 1.2)
ax.text(-0.06, 0, "FMCW", color=CYAN, fontsize=17, fontweight="bold", ha="right", va="center",
        transform=ax.get_yaxis_transform())
ax.text(3.08, 0, "десятки мВт, безперервно\nпередає й чує одночасно",
        color=BODY, fontsize=15, va="center")
for ax in axes:
    ax.set_xlim(0, 3)
    ax.axis("off")
    ax.axhline(0, color=MUTED, lw=0.6, alpha=0.6, zorder=0)
fig.savefig(os.path.join(out_dir, "p13_waveforms.png"), dpi=250, transparent=True)
plt.close(fig)


# ---- Doppler sees only the radial component ----
def drone(ax, x, y, s, col):
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        ax.plot([x, x + dx * s], [y, y + dy * s * 0.55], color=col, lw=1.6, zorder=4)
        ax.add_patch(plt.Circle((x + dx * s, y + dy * s * 0.55), s * 0.42, fill=False, color=col, lw=1.3, zorder=4))
    ax.add_patch(plt.Circle((x, y), s * 0.32, color=col, zorder=5))


fig, ax = plt.subplots(figsize=(5.6, 2.75))
fig.patch.set_alpha(0)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.9)
ax.set_aspect("equal")
ax.axis("off")

R = np.array([0.7, 1.4])  # radar
D = np.array([6.3, 2.5])  # drone
los = (D - R) / np.linalg.norm(D - R)
perp = np.array([-los[1], los[0]])
# beam cone and line of sight
half = np.radians(9)
rot = lambda v, a: np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])
L = 9.0
ax.add_patch(Polygon([R, R + L * rot(los, half), R + L * rot(los, -half)], closed=True,
                     color=CYAN, alpha=0.08, lw=0))
ax.plot(*zip(R, R + 8.8 * los), color=CYAN, lw=1, ls=(0, (4, 3)), alpha=0.8)
# radar icon
ax.add_patch(Polygon([R + [-0.35, -0.45], R + [0.35, -0.45], R + [0, 0.1]], closed=True, color=MUTED))
ax.add_patch(plt.Circle(R + [0, 0.12], 0.17, color=FG))
ax.text(R[0], R[1] - 0.75, "радар", color=MUTED, fontsize=15, ha="center", va="top")

drone(ax, *D, 0.38, FG)
theta = np.radians(55)  # angle between v and the line of sight
v = 1.9 * (np.cos(theta) * -los + np.sin(theta) * perp)  # moving towards the radar and sideways
vr = np.dot(v, -los) * -los
vp = v - vr
kw = dict(arrowstyle="-|>", mutation_scale=16, shrinkA=0, shrinkB=0, zorder=6)
ax.add_patch(FancyArrowPatch(D, D + v, color=FG, lw=2.2, **kw))
ax.add_patch(FancyArrowPatch(D, D + vr, color=CYAN, lw=2.6, **kw))
ax.add_patch(FancyArrowPatch(D, D + vp, color=MUTED, lw=1.6, ls="--", **kw))
ax.plot(*zip(D + vr, D + v), color=MUTED, lw=0.8, ls=":")
ax.plot(*zip(D + vp, D + v), color=MUTED, lw=0.8, ls=":")
# angle arc between -los and v
a0 = np.degrees(np.arctan2(-los[1], -los[0]))
a1 = np.degrees(np.arctan2(v[1], v[0]))
da = (a1 - a0 + 180) % 360 - 180  # signed shortest turn from -los to v
arc = np.radians(a0 + np.linspace(0, da, 30))
ax.plot(D[0] + 0.6 * np.cos(arc), D[1] + 0.6 * np.sin(arc), color=FG, lw=1)
am = np.radians(a0 + da / 2)
ax.text(D[0] + 0.85 * np.cos(am), D[1] + 0.85 * np.sin(am), "θ", color=FG, fontsize=18, ha="center", va="center")

ax.text(*(D + v + [0.1, 0.12]), r"$\mathbf{v}$", color=FG, fontsize=21, ha="left", va="bottom")
ax.text(*(D + vr + [-0.05, -0.3]), r"$\mathbf{v_r}$", color=CYAN, fontsize=21, ha="center", va="top")
ax.text(*(D + vp + [0.15, 0]), r"$v_\perp$", color=MUTED, fontsize=20, ha="left", va="center")
ax.text(9.9, 0.5, "Doppler бачить лише $v_r = v\\cdot\\cos\\,θ$\nθ = 90° → $v_r = 0$: дрон «нерухомий»",
        color=BODY, fontsize=16, ha="right", va="center", linespacing=1.4)
fig.savefig(os.path.join(out_dir, "p13_doppler_vr.png"), dpi=300, transparent=True)
plt.close(fig)
print("ok")
