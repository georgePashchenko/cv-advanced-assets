"""Slide 18: ego-velocity from one radar cycle on real data (RadarScenes, Schumann et al. 2021, CC BY-NC-SA 4.0).

Usage: p18_radar_egomotion.py <radar_data.h5> <sensors.json> <t_us> <out.png>
Takes the scan of each of the 4 radars nearest to t_us, maps detections to the vehicle frame
(azimuth + mounting yaw), fits v_r = -(cos φ, sin φ)·v with RANSAC + least squares, and compares the
result with the vehicle odometry. Points that are off the cosine are the moving targets.
"""
import json
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

h5, sensors_json, t_us, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
plt.rcParams["font.family"] = "Segoe UI"
BG, FG, MUTED, GRID = "#181a19", "#e8ecea", "#a0a6a3", "#2c302e"
TEAL, ORANGE = "#2ec4b6", "#f07a1e"
STATIC = 11  # RadarScenes label "static"

f = h5py.File(h5, "r")
r, odo = f["radar_data"][:], f["odometry"][:]
mount = {s["id"]: s for s in json.load(open(sensors_json)).values()}

# one scan per sensor: the timestamp of that sensor closest to t_us
rows = []
for sid in sorted(mount):
    ts = np.unique(r["timestamp"][r["sensor_id"] == sid])
    t = ts[np.argmin(np.abs(ts - t_us))]
    rows.append(np.flatnonzero((r["timestamp"] == t) & (r["sensor_id"] == sid)))
d = r[np.concatenate(rows)]
span_ms = (d["timestamp"].max() - d["timestamp"].min()) / 1e3
phi = d["azimuth_sc"] + np.array([mount[s]["yaw"] for s in d["sensor_id"]])  # vehicle frame
vr = d["vr"]
i = np.argmin(np.abs(odo["timestamp"] - t_us))
vx_odo, yaw_rate = float(odo["vx"][i]), float(odo["yaw_rate"][i])

A = -np.c_[np.cos(phi), np.sin(phi)]
rng = np.random.default_rng(0)
best = None
for _ in range(500):  # 2 points per hypothesis
    j = rng.choice(len(vr), 2, replace=False)
    try:
        v = np.linalg.solve(A[j], vr[j])
    except np.linalg.LinAlgError:
        continue
    inl = np.abs(A @ v - vr) < 0.3
    if best is None or inl.sum() > best.sum():
        best = inl
inl = best
v = np.linalg.lstsq(A[inl], vr[inl], rcond=None)[0]
moving_gt = d["label_id"] != STATIC
print(f"{len(vr)} dets in {span_ms:.0f} ms; v = ({v[0]:.2f}, {v[1]:.2f}) m/s; odometry vx = {vx_odo:.2f}, "
      f"yaw rate {yaw_rate:.3f} rad/s; outliers {(~inl).sum()}, of them labelled moving {(~inl & moving_gt).sum()}; "
      f"labelled moving total {moving_gt.sum()}")

fig, ax = plt.subplots(figsize=(6.3, 4.1), facecolor=BG)
fig.subplots_adjust(left=0.12, right=0.98, top=0.86, bottom=0.15)
ax.set_facecolor(BG)
deg = np.degrees(phi)
ax.scatter(deg[inl], vr[inl], s=9, color=FG, alpha=0.75, linewidths=0, label="статичні (inliers)", zorder=3)
mv, other = ~inl & moving_gt, ~inl & ~moving_gt
ax.scatter(deg[mv], vr[mv], s=34, marker="x", color=ORANGE, linewidths=1.6,
           label=f"викиди: рухомі цілі за розміткою ({mv.sum()} з {moving_gt.sum()})", zorder=4)
ax.scatter(deg[other], vr[other], s=26, marker="x", color=MUTED, linewidths=1.2,
           label=f"викиди: шум, відбиття ({other.sum()})", zorder=4)
g = np.linspace(deg.min() - 3, deg.max() + 3, 300)
gr = np.radians(g)
ax.plot(g, -(np.cos(gr) * v[0] + np.sin(gr) * v[1]), color=TEAL, lw=2.4, zorder=2,
        label=f"RANSAC + МНК: v = ({v[0]:.2f}, {v[1]:.2f}) м/с")
ax.set_xlabel("азимут детекції в системі авто φ, °", color=MUTED, fontsize=11.5)
ax.set_ylabel("v_r, м/с", color=MUTED, fontsize=11.5)
ax.tick_params(colors=MUTED, labelsize=10.5)
ax.grid(color=GRID, lw=0.8)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.set_title(f"RadarScenes · 4 радари 77 ГГц · {len(vr)} детекцій за {span_ms:.0f} мс\n"
             f"одометрія авто: {vx_odo:.2f} м/с", color=FG, fontsize=12, pad=8)
ax.legend(loc="lower left", fontsize=10, frameon=True, facecolor=BG, edgecolor=GRID, labelcolor=FG)
fig.savefig(out, dpi=220, facecolor=BG)
print("saved", out)
