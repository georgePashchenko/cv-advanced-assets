"""Slide 17: detection range vs RCS, R_max ∝ σ^(1/4), relative to a car (σ = 10 m²).

Usage: p17_rcs.py <out.png>
RCS values are orders of magnitude (they depend strongly on aspect and material), as in the lecture notes.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
plt.rcParams["font.family"] = "Segoe UI"
BG, FG, MUTED = "#181a19", "#e8ecea", "#a0a6a3"
GREY, CYAN, ORANGE = "#6b7280", "#38bdf8", "#f2a12a"

TARGETS = [  # name, sigma m², colour
    ("Легковик", 10, GREY),
    ("Людина", 1, GREY),
    ("Птах", 0.01, ORANGE),
    ("Малий мультикоптер", 0.01, CYAN),
    ("Мікродрон", 0.001, CYAN),
]
SIG_REF = 10.0

fig, ax = plt.subplots(figsize=(6.0, 4.42), facecolor=BG)
fig.subplots_adjust(left=0.36, right=0.97, top=0.80, bottom=0.13)
ax.set_facecolor(BG)
y = list(range(len(TARGETS)))[::-1]
for yi, (name, sig, col) in zip(y, TARGETS):
    r = (sig / SIG_REF) ** 0.25
    ax.barh(yi, r, height=0.62, color=col)
    ax.text(r + 0.02, yi, f"{r:.2f}", color=FG, fontsize=15, va="center", fontweight="bold")
    ax.text(-0.03, yi + 0.13, name, color=FG, fontsize=14.5, ha="right", va="center",
            transform=ax.get_yaxis_transform())
    ax.text(-0.03, yi - 0.24, f"σ ≈ {sig:g} м²", color=MUTED, fontsize=12, ha="right", va="center",
            transform=ax.get_yaxis_transform())
ax.set_xlim(0, 1.15)
ax.set_ylim(-0.6, len(TARGETS) - 0.4)
ax.set_yticks([])
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(colors=MUTED, labelsize=12)
ax.set_xlabel("дальність виявлення, частка від авто", color=MUTED, fontsize=13)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
fig.text(0.03, 0.94, r"Рівняння радара:  $R_{max} \propto \sigma^{1/4}$", color=CYAN, fontsize=17, fontweight="bold", va="center")
fig.text(0.03, 0.865, "той самий радар бачить малий дрон у 5.6 раза ближче, ніж авто",
         color=FG, fontsize=13, va="center")
fig.savefig(out, dpi=220, facecolor=BG)
print("saved", out)
