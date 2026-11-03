"""Slide 5: one KITTI object frame as camera / LiDAR / radar output, same crop for all three.

Usage: p5_sensor_outputs.py <kitti training dir> <frame> <u0> <out_dir>
Writes p5_camera.png, p5_lidar.png, p5_radar.png (aspect matches the slide boxes, 2.13:1).
Radar detections are simulated from label_2 boxes: KITTI has no radar.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

root, frame, u0, out_dir = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
stem = f"{frame:06d}"
rng = np.random.default_rng(5)

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 11})
FG, MUTED = "#e8ecea", "#9aa09d"

# --- data -------------------------------------------------------------------
img = np.asarray(Image.open(root / "image_2" / f"{stem}.png").convert("RGB"))
H, W = img.shape[:2]
pts = np.fromfile(root / "velodyne" / f"{stem}.bin", dtype=np.float32).reshape(-1, 4)
calib = {}
for ln in open(root / "calib" / f"{stem}.txt"):
    if ":" in ln:
        k, v = ln.split(":", 1)
        calib[k] = np.array(v.split(), float)
P2 = calib["P2"].reshape(3, 4)
R0 = np.eye(4); R0[:3, :3] = calib["R0_rect"].reshape(3, 3)
Tv = np.eye(4); Tv[:3] = calib["Tr_velo_to_cam"].reshape(3, 4)
P = P2 @ R0 @ Tv

Y = np.c_[pts[:, :3], np.ones(len(pts))] @ P.T
z = Y[:, 2]
with np.errstate(divide="ignore", invalid="ignore"):
    u, v = Y[:, 0] / z, Y[:, 1] / z
in_img = (z > 0.5) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
rng_m = np.linalg.norm(pts[:, :3], axis=1)
print(f"LiDAR: {len(pts)} points in scan, {in_img.sum()} in camera FOV")

objs = []
for ln in open(root / "label_2" / f"{stem}.txt"):
    f = ln.split()
    if f[0] in ("Car", "Van", "Truck", "Pedestrian", "Cyclist"):
        objs.append(dict(cls=f[0], box=np.array(f[4:8], float), hwl=np.array(f[8:11], float),
                         loc=np.array(f[11:14], float), ry=float(f[14]), occl=int(f[2])))

# Crop matching the slide box aspect
CW = round(H * 2.13)
x0, x1 = u0, u0 + CW
crop = img[:, x0:x1]


def new_fig():
    fig = plt.figure(figsize=(CW / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, CW); ax.set_ylim(H, 0); ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(out_dir, name), dpi=250, facecolor=fig.get_facecolor())
    plt.close(fig)


# The slide shows each image ~450 px wide (1/4 of the render), so text is sized for that
TAG_FS, LBL_FS = 15, 13


def tag(ax, text, color=FG):
    ax.text(10, H - 10, text, color=color, fontsize=TAG_FS, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#000000c0", ec="none"))


# --- camera -----------------------------------------------------------------
fig, ax = new_fig()
ax.imshow(crop, extent=(0, CW, H, 0), interpolation="lanczos")
# pixel-grid magnifier on the white car's tail light / bumper edge
pu, pv, n = 270, 206, 8
patch = crop[pv:pv + n, pu:pu + n]
ax.add_patch(Rectangle((pu, pv), n, n, fill=False, ec="#2dd4bf", lw=2))
ins = fig.add_axes([0.745, 0.30, 0.235, 0.66])
ins.imshow(patch, interpolation="nearest")
ins.set_xticks(np.arange(-0.5, n, 1)); ins.set_yticks(np.arange(-0.5, n, 1))
ins.grid(color="#000000aa", lw=1.0); ins.tick_params(length=0, labelbottom=False, labelleft=False)
for s in ins.spines.values():
    s.set_color("#2dd4bf"); s.set_linewidth(2.5)
ax.annotate("", xy=(0.745 * CW, 0.62 * H), xytext=(pu + n, pv + n / 2),
            arrowprops=dict(arrowstyle="-", color="#2dd4bf", lw=1.5))
r, g, b = patch[n // 2, n // 2]
ins.text(0.5, -0.03, f"({r}, {g}, {b})", transform=ins.transAxes, color=FG,
         fontsize=LBL_FS, ha="center", va="top", bbox=dict(boxstyle="round,pad=0.25", fc="#000000c0", ec="none"))
tag(ax, f"{W} × {H} px · RGB")
save(fig, "p5_camera.png")

# --- LiDAR ------------------------------------------------------------------
fig, ax = new_fig()
fig.set_facecolor("#05070a")
m = in_img & (u >= x0) & (u < x1)
order = np.argsort(-z[m])  # far first, near on top
cmap = plt.get_cmap("turbo")
sc = ax.scatter(u[m][order] - x0, v[m][order], c=np.clip(z[m][order], 0, 60), cmap=cmap, vmin=3, vmax=50,
                s=4.5, linewidths=0)
cax = fig.add_axes([0.62, 0.80, 0.35, 0.06])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
cb.set_ticks([5, 20, 35, 50]); cb.ax.tick_params(colors=FG, labelsize=LBL_FS, length=3)
cb.outline.set_visible(False)
cax.set_title("дальність, м", color=FG, fontsize=LBL_FS, pad=4)
tag(ax, f"{m.sum():,} точок на цій ділянці".replace(",", " "))
save(fig, "p5_lidar.png")

# --- radar (simulated) ------------------------------------------------------
fig, ax = new_fig()
fig.set_facecolor("#05070a")
dim = (crop.astype(float).mean(axis=2, keepdims=True) * 0.28).repeat(3, axis=2).astype(np.uint8)
ax.imshow(dim, extent=(0, CW, H, 0))
RADAR_H = 0.5  # m above ground, typical bumper mount

def radar_point(R, th):
    """Detection (range, azimuth) drawn at radar mount height; radar has no elevation."""
    X = np.array([R * np.sin(th), 1.65 - RADAR_H, R * np.cos(th), 1.0])  # camera frame, y down
    uu, vv, ww = P2 @ X
    return uu / ww - x0, vv / ww


dets = []  # (u, v, R, theta_deg, object index or -1 for clutter)
for i, o in enumerate(objs):
    h, w, l = o["hwl"]
    cx, _, cz = o["loc"]
    if cz < 2 or cz > 70 or o["occl"] == 2:  # 2 = largely occluded; 3 = unknown in KITTI
        continue
    # 2-3 returns from the face nearest to the sensor; 77 GHz noise: sigma_R 0.1 m, sigma_theta 0.8 deg
    near_x = cx - np.sign(cx) * w / 2 if abs(cx) > w / 2 else cx
    for _ in range(1 if o["cls"] in ("Pedestrian", "Cyclist") else 2 + int(rng.random() < 0.5)):
        px, pz = near_x + rng.normal(0, 0.15), cz - l / 2 + rng.uniform(0, 1.2)
        R = np.hypot(px, pz) + rng.normal(0, 0.1)
        th = np.arctan2(px, pz) + np.deg2rad(rng.normal(0, 0.8))
        dets.append((*radar_point(R, th), R, np.rad2deg(th), i))
for R, th in ((27.0, np.deg2rad(4.0)), (11.0, np.deg2rad(13.0))):  # clutter: manhole cover, kerb
    dets.append((*radar_point(R, th), R, np.rad2deg(th), -1))

shown = [d for d in dets if 0 <= d[0] < CW and 0 <= d[1] < H]
for du, dv, R, th, i in shown:
    print(f"  det u={du:6.0f} v={dv:5.0f} R={R:5.1f} th={th:+5.1f}  obj={i} "
          f"{objs[i]['cls'] + ' z=' + str(objs[i]['loc'][2]) if i >= 0 else 'clutter'}")
    ax.scatter([du], [dv], s=150, c="#f97316", edgecolors="#ffe1c9", linewidths=1.5, zorder=3)
# One label per object for the three nearest objects in view
first = {}
for d in sorted(shown, key=lambda d: d[2]):
    if d[4] >= 0 and d[4] not in first:
        first[d[4]] = d
for du, dv, R, th, _ in list(first.values())[:3]:
    right = du < 0.75 * CW  # label beside the dot, flipped near the right edge
    ax.annotate(f"{R:.1f} м · {th:+.0f}°", (du, dv), xytext=(12 if right else -12, 0), textcoords="offset points",
                color=FG, fontsize=LBL_FS, ha="left" if right else "right", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="#000000c8", ec="#f97316", lw=1.2))
tag(ax, f"{len(shown)} детекцій · симуляція з розмітки KITTI", "#ffd2b0")
save(fig, "p5_radar.png")
print("radar detections shown:", len(shown))
