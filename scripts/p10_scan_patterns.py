"""Slide 10 visual: LiDAR scan patterns in azimuth x elevation, colored by time within a frame.

Usage: p10_scan_patterns.py <out_dir>
Writes p10_scan_patterns.png (all five panels) and p10_scan_<key>.png (one per card).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

out_dir = sys.argv[1]
plt.rcParams["font.family"] = "Segoe UI"

BG = "#1c1d21"
FG = "#e8e8e8"
MUTED = "#8a8d93"
# plasma without its near-black start, so t = 0 stays visible on the dark slide
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "plasma_bright", plt.get_cmap("plasma")(np.linspace(0.22, 1.0, 256)))
T_FRAME = 100.0  # ms, 10 Hz

rng = np.random.default_rng(11)


def mech_spinning():
    # HDL-64E: 64 rings, +2..-24.8 deg (denser near horizon), 0.18 deg azimuth at 10 Hz
    upper = np.linspace(2.0, -8.33, 32)
    lower = np.linspace(-8.83, -24.8, 32)
    elev = np.concatenate([upper, lower])[::2]  # every 2nd ring for legibility
    az = np.arange(-180, 180, 1.8)
    A, E = np.meshgrid(az, elev)
    t = (A + 180) / 360 * T_FRAME
    return A.ravel(), E.ravel(), t.ravel()


def digital_spinning():
    # Ouster OS1-128: uniform 128 channels over 45 deg
    elev = np.linspace(22.5, -22.5, 128)[::4]
    az = np.arange(-180, 180, 1.8)
    A, E = np.meshgrid(az, elev)
    t = (A + 180) / 360 * T_FRAME
    return A.ravel(), E.ravel(), t.ravel()


def mems():
    # Small mirror raster: ~120 x 25 deg, fast horizontal zigzag, slow vertical sweep
    rows = np.linspace(12.5, -12.5, 26)
    az = np.linspace(-60, 60, 70)
    A, E, T = [], [], []
    for i, e in enumerate(rows):
        a = az if i % 2 == 0 else az[::-1]
        A.append(a)
        E.append(np.full_like(a, e))
        T.append(np.linspace(i, i + 1, len(a)) / len(rows) * T_FRAME)
    return np.concatenate(A), np.concatenate(E), np.concatenate(T)


def flash():
    # One pulse illuminates the whole (small) FOV: every pixel shares t = 0
    A, E = np.meshgrid(np.linspace(-30, 30, 48), np.linspace(-18, 18, 30))
    return A.ravel(), E.ravel(), np.zeros(A.size)


def livox(duration_ms):
    # Non-repetitive pattern (illustrative): incommensurate sweeps over 360 x 59 deg (-7..52)
    n = int(duration_ms * 60)
    t = np.linspace(0, duration_ms, n)
    s = t / 1000.0
    az = ((s * 360 * 10 + 40 * np.sin(2 * np.pi * 37.3 * s)) % 360) - 180
    el = 22.5 + 29.5 * np.sin(2 * np.pi * 241.8034 * s + 2.1 * np.sin(2 * np.pi * 7.07 * s))
    return az, el, t


PANELS = [
    ("mech", "Механічний spinning", "HDL-64E · 64 кільця · 26.8°", "#2ec4a6", mech_spinning,
     (-180, 180, -27, 5), "час росте з азимутом → перекіс"),
    ("digital", "Цифровий spinning", "OS1 · 128 каналів · 45°", "#3a9fe6", digital_spinning,
     (-180, 180, -25, 25), "той самий перекіс, рівномірні кільця"),
    ("mems", "MEMS / дзеркала", "растр ~120° × 25°", "#b26cf0", mems,
     (-70, 70, -16, 16), "рядок за рядком → перекіс по вертикалі"),
    ("flash", "Flash", "один спалах ~60° × 36°", "#f2a12a", flash,
     (-40, 40, -22, 22), "усі точки в t = 0 → без перекосу"),
    ("livox", "Неповторюваний", "Livox Mid-360 · 360° × 59°", "#f2702a", None,
     (-180, 180, -10, 55), "сірим — покриття за 1 с"),
]


def draw(ax, panel, compact=False):
    key, title, sub, color, fn, (x0, x1, y0, y1), note = panel
    ax.set_facecolor(BG)
    if key == "livox":
        a1, e1, _ = livox(1000)
        ax.scatter(a1, e1, s=0.3, c="#5a5e67", linewidths=0, rasterized=True)
        a, e, t = livox(T_FRAME)
    else:
        a, e, t = fn()
    size = {"flash": 5, "mems": 5, "livox": 1.2}.get(key, 2.2)
    ax.scatter(a, e, s=size, c=t, cmap=CMAP, vmin=0, vmax=T_FRAME, linewidths=0, rasterized=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    for sp in ax.spines.values():
        sp.set_color(color)
        sp.set_linewidth(1.6)
    ax.tick_params(colors=MUTED, labelsize=8, length=2)
    ax.set_xlabel("азимут, °", color=MUTED, fontsize=8, labelpad=1)
    ax.set_ylabel("кут місця, °", color=MUTED, fontsize=8, labelpad=1)
    if not compact:
        ax.set_title(f"{title}\n", color=color, fontsize=12, fontweight="bold", loc="left", pad=2)
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=FG, fontsize=8.5, va="bottom")
    ax.text(0.5, -0.33 if not compact else -0.3, note, transform=ax.transAxes, color=FG,
            fontsize=9, ha="center", va="top")


def colorbar(fig, rect):
    cax = fig.add_axes(rect)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(0, T_FRAME))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 25, 50, 75, 100])
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=2)
    cb.outline.set_edgecolor(MUTED)
    cb.set_label("час точки всередині кадру, мс (10 Гц)", color=FG, fontsize=9)


# Composite: five panels in a row
fig, axes = plt.subplots(1, 5, figsize=(16, 3.9), facecolor=BG)
fig.subplots_adjust(left=0.04, right=0.99, top=0.80, bottom=0.38, wspace=0.32)
for ax, panel in zip(axes, PANELS):
    draw(ax, panel)
colorbar(fig, [0.35, 0.13, 0.30, 0.035])
fig.savefig(os.path.join(out_dir, "p10_scan_patterns.png"), dpi=200, facecolor=BG)
plt.close(fig)

# One small image per card
for panel in PANELS:
    fig, ax = plt.subplots(figsize=(3.4, 1.9), facecolor=BG)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.95, bottom=0.36)
    draw(ax, panel, compact=True)
    fig.savefig(os.path.join(out_dir, f"p10_scan_{panel[0]}.png"), dpi=220, facecolor=BG)
    plt.close(fig)

print("ok")
