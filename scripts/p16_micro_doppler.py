"""Slides 16 / 54: micro-Doppler spectrograms of a hovering drone and a flying bird at 77 GHz, Tc = 50 us.

Usage: p16_micro_doppler.py <out.png>
The slow-time signal is sampled once per chirp (PRF 20 kHz -> unambiguous +-10 kHz), so blade returns
(tip 88 m/s -> 45 kHz) alias across the band exactly as on the slide.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

out = sys.argv[1]
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 13})
BG, FG, MUTED = "#181a19", "#e8ecea", "#a0a6a3"
rng = np.random.default_rng(3)

LAM = 3e8 / 77e9            # 3.9 mm
PRF = 1 / 50e-6             # one sample per chirp
k = 4 * np.pi / LAM         # phase per metre of radial displacement


def noise(n, level):
    return level * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)


# --- drone: body ~0 m/s, one rotor (2 blades), 7000 rpm, r up to 12 cm --------------------------
T_D = 0.1
t = np.arange(int(T_D * PRF)) / PRF
omega = 2 * np.pi * 7000 / 60
# A blade flashes when it is perpendicular to the line of sight: twice per revolution per blade pair.
# The flash (~lambda / (2 L omega) ~ 20 us) is shorter than one chirp, so it lands in a single chirp
# as a broadband impulse that covers the whole +-10 kHz band after aliasing.
blades = np.zeros(len(t), complex)
flash_rate = 2 * 7000 / 60
for tf in np.arange(rng.uniform(0, 1 / flash_rate), T_D, 1 / flash_rate):
    blades[min(int(round(tf * PRF)), len(t) - 1)] += 1.2 * np.exp(2j * np.pi * rng.random())
drift = 0.15 * np.sin(2 * np.pi * 0.8 * t)                     # hover wobble, cm/s level
sig_d = np.exp(1j * k * np.cumsum(drift) / PRF) + blades + noise(len(t), 0.02)
v_tip = omega * 0.12
print(f"tip {v_tip:.0f} m/s -> {2 * v_tip / LAM / 1e3:.1f} kHz, flashes {2 * 7000 / 60:.0f}/s per rotor")

# --- bird: body 8 m/s towards radar, wings 0.3 m flapping 6 Hz, +-40 deg -------------------------
T_B = 1.0
tb = np.arange(int(T_B * PRF)) / PRF
v_body = 8.0
sig_b = 1.0 * np.exp(1j * k * v_body * tb)
flap = np.deg2rad(40) * np.sin(2 * np.pi * 6 * tb)
for side in (-1, 1):
    for r in np.linspace(0.03, 0.3, 30):
        # LOS 30 deg below the flapping plane: half of the vertical wing motion is radial
        sig_b += 0.05 * np.exp(1j * k * (v_body * tb + 0.5 * r * np.sin(flap)))
sig_b += noise(len(tb), 0.02)


def spectrogram(x, nper):
    f, tt, Z = stft(x, fs=PRF, window="hann", nperseg=nper, noverlap=nper - max(1, nper // 16),
                    return_onesided=False)
    S = 20 * np.log10(np.abs(np.fft.fftshift(Z, axes=0)) + 1e-9)
    return np.fft.fftshift(f) / 1e3, tt, S - S.max()


fig, axs = plt.subplots(1, 2, figsize=(9.9, 6.16), facecolor=BG)
fig.subplots_adjust(left=0.085, right=0.985, top=0.87, bottom=0.22, wspace=0.18)
panels = [(axs[0], sig_d, 32, "дрон, зависання · 1 гвинт, 7000 об/хв", T_D, "час, мс", 1e3),
          (axs[1], sig_b, 256, "птах, 8 м/с · махи крил 6 Гц", T_B, "час, с", 1)]
for ax, x, nper, title, T, xl, sc in panels:
    f, tt, S = spectrogram(x, nper)
    ax.pcolormesh(tt * sc, f, S, shading="auto", cmap="magma", vmin=-40, vmax=0, rasterized=True)
    ax.set_facecolor(BG)
    ax.set_xlim(0, T * sc); ax.set_ylim(-10, 10)
    ax.set_title(title, color=FG, fontsize=13.5, pad=8)
    ax.set_xlabel(xl, color=MUTED)
    ax.tick_params(colors=MUTED)
    for sp in ax.spines.values():
        sp.set_color("#3a3f3c")
axs[0].set_ylabel("Doppler, кГц", color=MUTED)
axs[1].annotate("тіло: 2v/λ ≈ 4.1 кГц", (0.5, 4.1), xytext=(0.5, 8.6), color=FG, fontsize=12, ha="center",
                arrowprops=dict(arrowstyle="->", color=FG, lw=1))
axs[0].annotate("тіло ≈ 0", (50, 0), xytext=(50, -8.6), color=FG, fontsize=12, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="#000000b0", ec="none"),
                arrowprops=dict(arrowstyle="->", color=FG, lw=1))
fig.text(0.5, 0.035, "модель: 77 ГГц, Tc = 50 мкс → ±10 кГц. Лопаті (до 45 кГц) загортаються у вертикальні спалахи:\n"
         "233 за секунду на гвинт, 4 гвинти — вчетверо щільніше. Крила — синусоїда навколо лінії тіла.",
         color=MUTED, fontsize=10.5, ha="center")
fig.savefig(out, dpi=200, facecolor=BG)
print("saved", out)
