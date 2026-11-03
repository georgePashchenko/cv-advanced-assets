"""Slide 23: LiDAR -> camera projection on KITTI with correct extrinsics vs yaw +0.5 deg vs 5 cm shift.

Usage: p23_calib_validation.py <kitti training dir> <out.png>
Rows: a near edge (~6 m) and a far car (~33 m); columns: OK / yaw / shift. Shifts are measured, not drawn.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

root, out = Path(sys.argv[1]), sys.argv[2]
stem = "000008"
plt.rcParams.update({"font.family": "Segoe UI"})
BG, FG, MUTED = "#181a19", "#e8ecea", "#9aa09d"
TEAL, ORANGE, PURPLE = "#2dd4bf", "#f97316", "#c084fc"

img = np.asarray(Image.open(root / "image_2" / f"{stem}.png").convert("RGB"))
pts = np.fromfile(root / "velodyne" / f"{stem}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
cal = {}
for ln in open(root / "calib" / f"{stem}.txt"):
    if ":" in ln:
        k, v = ln.split(":", 1)
        cal[k] = np.array(v.split(), float)
P2 = cal["P2"].reshape(3, 4)
R0 = np.eye(4); R0[:3, :3] = cal["R0_rect"].reshape(3, 3)
Tv = np.eye(4); Tv[:3] = cal["Tr_velo_to_cam"].reshape(3, 4)
f = P2[0, 0]

Xc = (R0 @ Tv @ np.c_[pts, np.ones(len(pts))].T)[:3]  # rectified camera frame, metres
Xc = Xc[:, Xc[2] > 1.0]

YAW = np.deg2rad(0.5)
DT = 0.05
c, s = np.cos(YAW), np.sin(YAW)
Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # rotation about the camera's vertical (y) axis
CASES = [("калібровка OK", Xc, TEAL), ("yaw +0.5°", Ry @ Xc, ORANGE), ("зсув 5 см", Xc + np.array([[DT], [0], [0]]), PURPLE)]


def project(X):
    uvw = P2 @ np.vstack([X, np.ones(X.shape[1])])
    return uvw[0] / uvw[2], uvw[1] / uvw[2], X[2]


u0, v0, z0 = project(Xc)
# (label, window u0, v0, size, depth range, label_2 box of the object, which side is the edge we read)
WINDOWS = [("ближній край", 548, 178, 140, (4.0, 11.0), (334.85, 178.94, 624.50, 372.04), "right"),
           ("далека машина", 722, 146, 90, (28.0, 36.0), (741.18, 168.83, 792.25, 208.43), "left")]
NOT_GROUND = Xc[1] < 1.45  # camera is 1.65 m above the road, y points down

fig = plt.figure(figsize=(12, 8.2), facecolor=BG)
gs = fig.add_gridspec(2, 3, left=0.085, right=0.99, top=0.88, bottom=0.03, wspace=0.05, hspace=0.16)
cmap = plt.get_cmap("turbo")

for r, (wlabel, wu, wv, ws, (zmin, zmax), (bx0, by0, bx1, by1), side) in enumerate(WINDOWS):
    win = (u0 >= wu) & (u0 < wu + ws) & (v0 >= wv) & (v0 < wv + ws)
    in_box = (u0 >= bx0) & (u0 <= bx1) & (v0 >= by0) & (v0 <= by1)
    obj = win & in_box & NOT_GROUND & (z0 > zmin) & (z0 < zmax)
    Z = float(np.median(z0[obj]))
    theta = np.arctan((np.median(u0[obj]) - P2[0, 2]) / f)
    edge = np.percentile(u0[obj], 97 if side == "right" else 3)  # object edge with correct extrinsics
    for col, (clabel, X, accent) in enumerate(CASES):
        ax = fig.add_subplot(gs[r, col])
        ax.imshow(img[wv:wv + ws, wu:wu + ws], extent=(wu, wu + ws, wv + ws, wv), interpolation="lanczos")
        u, v, z = project(X)
        m = (u >= wu - 10) & (u < wu + ws + 10) & (v >= wv) & (v < wv + ws)
        on = m & obj  # same physical points as the object mask, wherever they now project
        ax.scatter(u[m & ~obj], v[m & ~obj], s=6, c="#d0d4d2", alpha=0.45, linewidths=0)
        ax.scatter(u[on], v[on], s=26, c=accent, edgecolors="black", linewidths=0.5)
        ax.axvline(edge, color="white", lw=1.6, ls=(0, (4, 3)), alpha=0.9)
        ax.set_xlim(wu, wu + ws); ax.set_ylim(wv + ws, wv)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(accent); sp.set_linewidth(2.2)
        if r == 0:
            ax.set_title(clabel, color=accent, fontsize=26, fontweight="bold", pad=8)
        if col == 0:
            ax.set_ylabel(f"{wlabel}\nZ ≈ {Z:.0f} м", color=FG, fontsize=21, labelpad=8)
            txt = "точки на краях"
        else:
            du = float(np.median(u[obj] - u0[obj]))
            pred = f * YAW / np.cos(theta) ** 2 if col == 1 else f * DT / Z
            formula = "f·δθ" if col == 1 else "f·δt/Z"
            txt = f"Δu = {du:.1f} px\n{formula} = {pred:.1f}"
            print(f"{wlabel:14s} Z={Z:5.1f} theta={np.rad2deg(theta):5.1f}deg {clabel:14s} du={du:5.2f} pred={pred:5.2f}")
        ax.text(0.5, 0.03, txt, transform=ax.transAxes, ha="center", va="bottom", color=FG, fontsize=20,
                bbox=dict(boxstyle="round,pad=0.3", fc="#000000c8", ec=accent if col else "none", lw=1.2))

fig.text(0.075, 0.962, f"KITTI {stem} · f = {f:.1f} px · пунктир = край об'єкта за правильної калібровки",
         color=MUTED, fontsize=17, va="center")
fig.savefig(out, dpi=200, facecolor=BG)
print("saved", out)
