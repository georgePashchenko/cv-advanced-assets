"""Lesson 13: illustrations and memes built from Gemini renders (no text in the renders).

Usage: l13_gen.py <repo_root>
Reads lesson13/gen/versions, lesson13/memes/versions and the lesson 12 platform renders in lesson12/gen2,
writes slide images to lesson13/gen and captioned memes to lesson13/memes.
The 32x32 crops on the "polygon" slide are illustrations of what the synthetic demo data imitates, not demo data.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = sys.argv[1]
sys.argv[1] = os.path.join(ROOT, "lesson12")           # l12_gen2 reads its lesson dir from argv on import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l12_gen2 import CARD, FONT, classic, row, tag    # noqa: E402

GEN, OUT = os.path.join(ROOT, "lesson13/gen/versions"), os.path.join(ROOT, "lesson13/gen")
MEME_SRC, MEME_OUT = os.path.join(ROOT, "lesson13/memes/versions"), os.path.join(ROOT, "lesson13/memes")
L12 = os.path.join(ROOT, "lesson12/gen2")


def render(name):
    return Image.open(os.path.join(GEN, name + "_v1.png")).convert("RGB")


def save(im, name):
    im.save(os.path.join(OUT, name), quality=92)
    print("saved", name, im.size)


def band(im, aspect, top):
    """Full-width crop of the given aspect starting at `top` (fraction of the height)."""
    W, H = im.size
    h = int(W / aspect)
    y = int(top * H)
    return im.crop((0, min(y, H - h), W, min(y, H - h) + h))


def crop32(im, cx, cy, half, size, shift=(0, 0)):
    """Square crop around (cx, cy) as the 32x32 px network input, shown enlarged with visible pixels."""
    c = im.crop((cx - half + shift[0], cy - half + shift[1], cx + half + shift[0], cy + half + shift[1]))
    return c.resize((32, 32), Image.LANCZOS).resize((size, size), Image.NEAREST)


def night(im):
    a = np.asarray(im).astype(float) / 255
    a = 0.25 * a ** 1.5 * np.array([0.8, 0.9, 1.0]) + np.random.default_rng(1).normal(0, 0.02, a.shape)
    return Image.fromarray((a.clip(0, 1) * 255).astype(np.uint8))


def sky(size):
    g = np.linspace(0, 1, size)[:, None, None]
    a = (1 - g) * np.array([70, 130, 200]) + g * np.array([170, 205, 235])
    return Image.fromarray(np.broadcast_to(a, (size, size, 3)).astype(np.uint8))


# ---------- slide 5: three ways to waste a month ----------
def p5():
    for name, out, top in (("h1_tuning", "p5_a.jpg", 0.0), ("h2_random", "p5_b.jpg", 0.03),
                           ("h3_wrong_place", "p5_c.jpg", 0.06)):
        save(band(render(name), 416 / 156, top), out)


# ---------- slide 6: what the synthetic polygon imitates ----------
def p6():
    drone, drone_t, bird, field = render("drone_rgb"), render("drone_t"), render("bird_rgb"), render("field")
    S = 560
    a, b = crop32(drone, 1024, 1024, 420, S), crop32(drone_t, 1024, 1024, 420, S)
    tag(a, (16, 14), "RGB", 44); tag(b, (16, 14), "LWIR", 44)
    save(row([a, b], 72), "p6_frame.jpg")
    s = 360
    save(row([crop32(drone, 420, 1500, 420, s), crop32(drone, 1024, 1024, 420, s), crop32(bird, 1024, 1024, 420, s)], 56),
         "p6_classes.jpg")
    f = 216                                             # five consecutive frames of one event: almost identical
    save(row([crop32(drone, 1024, 1024, 420, f, shift=(-18 * k, 6 * k)) for k in range(5)], 28), "p6_flight.jpg")
    t = 272
    tiles = [sky(t), crop32(drone, 420, 1500, 420, t), crop32(bird, 420, 420, 420, t), crop32(field, 1024, 1300, 420, t)]
    top, bottom = row(tiles, 34), row([night(x) for x in tiles], 34)
    grid = Image.new("RGB", (top.width, 2 * t + 34), CARD)
    grid.paste(top, (0, 0)); grid.paste(bottom, (0, t + 34))
    save(grid, "p6_meta.jpg")


# ---------- slide 15: the cloakroom token ----------
def p15():
    save(band(render("cloakroom"), 4.0, 0.33), "p15_cloakroom.jpg")


# ---------- platforms: strips for the platform slides and the grid for the section divider ----------
PLATFORMS = [("isr", "p38_isr.jpg", "01 НІЧНА ISR", 0.25), ("fpv", "p55_fpv.jpg", "02 FPV / ПЕРЕХОПЛЮВАЧ", 0.28),
             ("mast", "p56_mast.jpg", "03 ЩОГЛА C-UAS", 0.02), ("ugv", "p57_ugv.jpg", "04 UGV У ДИМУ", 0.30)]


def platforms():
    tiles = []
    for key, src, label, top in PLATFORMS:
        im = Image.open(os.path.join(L12, src)).convert("RGB")
        save(band(im, 416 / 130, top), f"p_{key}_strip.jpg")
        W, H = im.size
        w = int(H * 4 / 3)
        box = (220, 150, 1220, 900) if key == "isr" else ((W - w) // 2, 0, (W - w) // 2 + w, H)   # below the RGB / LWIR tags
        t = im.crop(box).resize((720, 540), Image.LANCZOS)
        tag(t, (20, 484), label, 30)
        tiles.append(t)
    gap = 28
    grid = Image.new("RGB", (2 * 720 + gap, 2 * 540 + gap), CARD)
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % 2) * (720 + gap), (i // 2) * (540 + gap)))
    save(grid, "p33_platforms.jpg")


# ---------- memes ----------
MEMES = [("m1_tag", "git tag ds-v2", "А dvc push хто робитиме?"),
         ("m2_params", "Експеримент 47 без dvc exp", "Які там були параметри?"),
         ("m3_anchors", "Grad-CAM для YOLO", "Для якого з 8400 якорів?")]


def memes():
    for name, top, bottom in MEMES:
        im = Image.open(os.path.join(MEME_SRC, name + "_v1.png")).convert("RGB").resize((1200, 896), Image.LANCZOS)
        classic(im, top, bottom).save(os.path.join(MEME_OUT, name + ".jpg"), quality=92)
    print("memes", len(MEMES))


if __name__ == "__main__":
    p5(); p6(); p15(); platforms(); memes()
