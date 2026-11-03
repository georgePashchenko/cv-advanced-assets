"""Slide 32: LiDAR -> RGB on KITTI frame 8, colour = log depth, with a parallax inset (per-pixel vs 5x5 z-buffer).

Usage: p32_kitti_projection.py <kitti training dir> <out.png>
Projection is the code from the previous slide (P2 @ R0 @ Tv, z > 0.5, FOV, np.minimum.at).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image
from scipy.ndimage import minimum_filter

root, out = Path(sys.argv[1]), sys.argv[2]
stem = "000008"
plt.rcParams.update({"font.family": "Segoe UI"})
FG, ORANGE, TEAL, MAGENTA = "#e8ecea", "#f97316", "#2dd4bf", "#ff3df2"

img = np.asarray(Image.open(root / "image_2" / f"{stem}.png").convert("RGB"))
H, W = img.shape[:2]
pts = np.fromfile(root / "velodyne" / f"{stem}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
cal = {}
for ln in open(root / "calib" / f"{stem}.txt"):
    if ":" in ln:
        k, v = ln.split(":", 1)
        cal[k] = np.array(v.split(), float)
P2 = cal["P2"].reshape(3, 4)
R0 = np.eye(4); R0[:3, :3] = cal["R0_rect"].reshape(3, 3)
Tv = np.eye(4); Tv[:3] = cal["Tr_velo_to_cam"].reshape(3, 4)

Y = (P2 @ R0 @ Tv @ np.c_[pts, np.ones(len(pts))].T).T
z = Y[:, 2]; m = z > 0.5
u, v, z = Y[m, 0] / z[m], Y[m, 1] / z[m], z[m]
m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
u, v, z = u[m], v[m], z[m]
ui, vi = u.astype(int), v.astype(int)
depth = np.full((H, W), np.inf)
np.minimum.at(depth, (vi, ui), z)
coverage = np.isfinite(depth).mean()

# Parallax: drop a point if a clearly closer point exists in its 5x5 neighbourhood
near5 = minimum_filter(depth, size=5, mode="constant", cval=np.inf)[vi, ui]
bleed = (z - near5) > np.maximum(1.0, 0.1 * z)
print(f"{len(z)} points in FOV, coverage {100 * coverage:.2f}%, 5x5 window drops {bleed.sum()}")

norm = LogNorm(4, 80)
cmap = plt.get_cmap("turbo")
fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow((img * 0.82).astype(np.uint8), extent=(0, W, H, 0))
order = np.argsort(-z)
sc = ax.scatter(u[order], v[order], c=z[order], cmap=cmap, norm=norm, s=1.3, linewidths=0)
ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

ax.text(8, H - 8, f"KITTI {stem} · {len(z):,} точок · {100 * coverage:.1f} % пікселів мають точку".replace(",", " "),
        color=FG, fontsize=12, va="bottom", bbox=dict(boxstyle="round,pad=0.3", fc="#000000c0", ec="none"))

cax = fig.add_axes([0.015, 0.86, 0.2, 0.035])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[5, 10, 20, 40, 80])
cb.ax.set_xticklabels(["5", "10", "20", "40", "80"])
cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())  # no 4x10^0-style minor labels
cb.ax.tick_params(colors=FG, labelsize=10, length=2)
cb.outline.set_visible(False)
cax.set_title("глибина, м (log)", color=FG, fontsize=10.5, pad=3, loc="left")

# Parallax inset: same patch, per-pixel z-buffer vs 5x5 window
pu, pv, ps = 552, 168, 80
ax.add_patch(Rectangle((pu, pv), ps, ps, fill=False, ec=FG, lw=1.2, ls=(0, (3, 2))))
sel = (u >= pu) & (u < pu + ps) & (v >= pv) & (v < pv + ps)
for k, (title, keep, accent) in enumerate((("попіксельно", np.ones_like(bleed), ORANGE),
                                            ("вікно 5×5", ~bleed, TEAL))):
    iw = 0.125
    ins = fig.add_axes([0.718 + k * 0.137, 0.955 - iw * W / H, iw, iw * W / H])  # square in pixels
    ins.imshow((img[pv:pv + ps, pu:pu + ps] * 0.85).astype(np.uint8), extent=(pu, pu + ps, pv + ps, pv))
    s = sel & keep
    ins.scatter(u[s & ~bleed], v[s & ~bleed], c=z[s & ~bleed], cmap=cmap, norm=norm, s=9, linewidths=0)
    if k == 0:
        ins.scatter(u[s & bleed], v[s & bleed], s=26, facecolors="none", edgecolors=MAGENTA, linewidths=1.0)
    ins.set_xlim(pu, pu + ps); ins.set_ylim(pv + ps, pv)
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_color(accent); sp.set_linewidth(2)
    ins.text(0.5, -0.04, title, transform=ins.transAxes, ha="center", va="top", color=FG, fontsize=11,
             bbox=dict(boxstyle="round,pad=0.25", fc="#000000c8", ec=accent, lw=1))
    if k == 0:
        ins.text(0.03, 0.97, "фон крізь край", transform=ins.transAxes, ha="left", va="top", color=MAGENTA,
                 fontsize=9.5, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="#000000c0", ec="none"))
        fig.add_artist(ConnectionPatch(xyA=(pu + ps, pv), coordsA=ax.transData, xyB=(0, 1), coordsB=ins.transAxes,
                                       color=FG, lw=0.9, ls=(0, (3, 2))))
        fig.add_artist(ConnectionPatch(xyA=(pu + ps, pv + ps), coordsA=ax.transData, xyB=(0, 0),
                                       coordsB=ins.transAxes, color=FG, lw=0.9, ls=(0, (3, 2))))

fig.savefig(out, dpi=250)
print("saved", out)
