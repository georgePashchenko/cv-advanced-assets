"""Lesson 12, second pass: illustrations for the text-only slides, memes, and slide mock-ups.

Usage: l12_gen2.py <lesson12_dir> [<thumbs_dir> <mock_dir>]
Reads Gemini renders (no text in them) from <lesson12_dir>/gen2/versions, memes/versions and scene/versions,
writes slide images to <lesson12_dir>/gen2 and captioned memes to <lesson12_dir>/memes. Everything that carries a
number (drone size in pixels, grey levels, box scores) is drawn here, not by Gemini.
With 1600x900 slide thumbnails (sNN.png) it also writes mock-ups of the slides with the images in place.
"""
import io
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

L12 = sys.argv[1]
GEN, OUT, SCENE = (os.path.join(L12, p) for p in ("gen2/versions", "gen2", "scene/versions"))
MEME_SRC, MEME_OUT = os.path.join(L12, "memes/versions"), os.path.join(L12, "memes")
FONT, IMPACT = "C:/Windows/Fonts/seguisb.ttf", "C:/Windows/Fonts/impact.ttf"
CARD, SLIDE, FG, MUTED = (30, 33, 36), (27, 29, 28), (232, 236, 234), (160, 166, 163)
BLUE, ORANGE, GREEN = (74, 158, 255), (240, 122, 30), (46, 196, 182)
rng = np.random.default_rng(12)


def load(path, gray=False):
    return np.asarray(Image.open(path).convert("L" if gray else "RGB")).astype(float) / 255


def to_img(a):
    a = np.clip(a, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8)).convert("RGB")


def resize(a, w, h=None, nearest=False):
    im = to_img(a)
    h = h or round(im.height * w / im.width)
    r = np.asarray(im.resize((w, h), Image.NEAREST if nearest else Image.LANCZOS)).astype(float) / 255
    return r[..., 0] if a.ndim == 2 else r


def gray(a):
    return a if a.ndim == 2 else a.mean(axis=2)


def stretch(t, lo=1, hi=99.7, gamma=1.4):
    a, b = np.percentile(t, [lo, hi])
    return np.clip((t - a) / (b - a), 0, 1) ** gamma


def palette(t):
    return plt.get_cmap("inferno")(stretch(t))[..., :3]


def night(rgb):
    """Visible camera at night without lighting: little signal, high gain -> dark and grainy."""
    g = 0.22 * rgb ** 1.6 * np.array([0.85, 1.0, 0.9])
    g += rng.normal(0, 0.035, g.shape[:2])[..., None] + rng.normal(0, 0.015, g.shape)
    return g.clip(0, 1)


def tag(im, xy, text, size, fill=FG):
    d = ImageDraw.Draw(im, "RGBA")
    f = ImageFont.truetype(FONT, size)
    x0, y0, x1, y1 = d.textbbox(xy, text, font=f)
    p = size // 3
    d.rounded_rectangle((x0 - p, y0 - p, x1 + p, y1 + p), radius=p, fill=(0, 0, 0, 165))
    d.text(xy, text, font=f, fill=fill)


def row(tiles, gap, bg=CARD):
    """Tiles (PIL images of equal height) side by side on the card colour."""
    w = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    canvas = Image.new("RGB", (w, tiles[0].height), bg)
    x = 0
    for t in tiles:
        canvas.paste(t, (x, 0)); x += t.width + gap
    return canvas


def save(im, name, q=92):
    im.save(os.path.join(OUT, name), quality=q)
    print("saved", name, im.size)
    return im


def person_box(t):
    """Bounding box of the walker = largest very bright blob of the night LWIR frame."""
    lab, n = ndimage.label(t > np.percentile(t, 99.3))
    k = np.argmax(ndimage.sum(np.ones_like(t), lab, range(1, n + 1))) + 1
    ys, xs = np.nonzero(lab == k)
    return xs.min(), ys.min(), xs.max(), ys.max()


# ---------- slide 8: emissivity, RGB / LWIR pair ----------
def p8():
    S, gap = 880, 60
    tiles = []
    for name, lab in (("p8_rgb_v1", "RGB"), ("p8_t_v1", "LWIR")):
        im = to_img(resize(load(os.path.join(GEN, name + ".png")), S, S))
        tag(im, (S // 28, S // 28), lab, S // 16)
        tiles.append(im)
    return save(row(tiles, gap), "p8_pair.jpg")


# ---------- slide 11: a 0.3 m drone on a 640x512 core with a 50 mm lens ----------
def p11():
    sky = resize(gray(load(os.path.join(GEN, "p11_sky_v1.png"))), 640, 512)
    x0, y0, w, h, up, ss = 300, 150, 44, 33, 12, 16
    base = sky[y0:y0 + h, x0:x0 + w]
    tiles = []
    for km, px in ((0.5, 2.5), (1, 1.25), (2, 0.6)):       # IFOV 0.24 mrad: 0.3 m / R / IFOV
        yy, xx = np.mgrid[:h * ss, :w * ss] / ss
        disc = (np.hypot(xx - (w / 2 + 0.3), yy - (h / 2 + 0.2)) < px / 2).astype(float)
        blob = ndimage.gaussian_filter(disc.reshape(h, ss, w, ss).mean(axis=(1, 3)), 0.45)   # pixel area, then optics
        f = base + 0.62 * blob + np.random.default_rng(3).normal(0, 0.012, base.shape)
        im = to_img(resize(np.clip((f - 0.12) / 0.75, 0, 1), w * up, h * up, nearest=True))
        tag(im, (14, 12), f"{km:g} км · {px:g} px", 40)
        tiles.append(im)
    return save(row(tiles, 14), "p11_drone.jpg")


# ---------- slide 14: the same frame along the signal chain ----------
def scene_crop(box=(560, 330, 2060, 1330), w=300):
    t = load(os.path.join(SCENE, "t_night_v1.png"), gray=True)
    x0, y0, x1, y1 = box
    return resize(t[y0:y1, x0:x1], w)


def labelled(a, text, w, h, size):
    """Tile with a caption above it."""
    canvas = Image.new("RGB", (w, h + int(size * 1.7)), CARD)
    canvas.paste(to_img(a).resize((w, h), Image.LANCZOS), (0, int(size * 1.7)))
    ImageDraw.Draw(canvas).text((0, int(size * 0.2)), text, font=ImageFont.truetype(FONT, size), fill=MUTED)
    return canvas


def p14():
    t = scene_crop()
    H, W = t.shape
    r = np.random.default_rng(5)
    counts = 0.44 + 0.11 * t                                     # 14-bit raw: the scene fills a sliver of the range
    raw = counts + r.normal(0, 0.012, W)[None, :] + r.normal(0, 0.006, (H, W))   # column offsets before NUC
    nuc = counts + r.normal(0, 0.002, (H, W))
    agc = stretch(nuc, 0.5, 99.9, 1.0)
    pal = plt.get_cmap("inferno")(agc)[..., :3]
    small = to_img(pal).resize((W // 2, H // 2), Image.LANCZOS)
    buf = io.BytesIO(); small.save(buf, "JPEG", quality=4)        # starved bitrate: blocks and lost detail
    h264 = np.asarray(Image.open(buf).convert("RGB").resize((W, H), Image.NEAREST)) / 255
    rad = to_img(np.repeat(nuc[..., None], 3, 2)).resize((600, 400), Image.LANCZOS)
    bx = person_box(t)
    k = 600 / W
    tag(rad, (int(bx[2] * k) + 10, int(bx[1] * k)), "31 °C", 44)
    tag(rad, (20, 320), "2 °C", 44)
    tiles = [("raw 14 біт", raw), ("після NUC", nuc), ("counts → °C", np.asarray(rad) / 255), ("AGC → 8 біт", agc),
             ("палітра (LUT)", pal), ("H.264", h264)]
    made = [labelled(a, s, 600, 360, 44) for s, a in tiles]
    return [save(row(made[i:i + 2], 48), f"p14_{'abc'[i // 2]}.jpg") for i in (0, 2, 4)]


# ---------- slide 17: 16-bit TIFF read as its high byte ----------
def p17():
    t = scene_crop((1250, 520, 2000, 1020), 375)                  # the walker and the road
    raw = 4096 + 7100 * stretch(t, 0.5, 99.9, 1.0)                # 14-bit counts spanning 28 high-byte steps
    hi = np.floor(raw / 256)                                      # im >> 8: 0…63
    own = np.round(stretch(t, 0.5, 99.9, 1.0) * 134) / 134        # own LO / HI mapping: 135 levels
    n = len(np.unique(hi))
    tiles = [("свій мапінг: 135 рівнів", own), (f">> 8: {n} рівнів (розтягнуто)", (hi - hi.min()) / np.ptp(hi)),
             (">> 8 як є: 0…63 з 255", hi / 255)]
    return save(row([labelled(a, s, 600, 360, 42) for s, a in tiles], 36), "p17_levels.jpg")


# ---------- slide 16: FFC shutter ----------
def p16():
    im = Image.open(os.path.join(GEN, "p16_ffc_v1.png")).convert("RGB")
    W, H = im.size
    a = np.asarray(im.crop((int(0.06 * W), int(0.07 * H), int(0.97 * W), int(0.93 * H)))).astype(float)
    h, w = a.shape[:2]
    corners = np.concatenate([a[:30, :30], a[:30, -30:], a[-30:, :30], a[-30:, -30:]]).reshape(-1, 3)
    a += np.array(SLIDE) - np.median(corners, axis=0)            # render background -> slide background
    yy = np.minimum(np.arange(h), np.arange(h)[::-1])[:, None]
    xx = np.minimum(np.arange(w), np.arange(w)[::-1])[None, :]
    m = np.clip(np.minimum(yy, xx) / 120, 0, 1)[..., None]       # fade the borders into the slide
    return save(Image.fromarray((m * a + (1 - m) * np.array(SLIDE)).clip(0, 255).astype(np.uint8)), "p16_ffc.jpg")


# ---------- slide 20: two cameras, baseline and fields of view ----------
def p20():
    im = Image.open(os.path.join(GEN, "p21_gimbal_v1.png")).convert("RGB")
    W, H = im.size
    im = im.crop((0, int(0.10 * H), W, int(0.10 * H) + int(W / 4.15)))
    s = im.height // 10
    tag(im, (int(0.30 * W), int(0.06 * im.height)), "RGB · 1280×720 · 60°", s, fill=GREEN)
    tag(im, (int(0.52 * W), int(0.42 * im.height)), "LWIR · 640×512 · 45°", s, fill=ORANGE)
    tag(im, (int(0.19 * W), int(0.80 * im.height)), "B = 5 см", s)
    return save(im, "p20_gimbal.jpg")


# ---------- slide 33: late fusion on boxes ----------
def p33():
    t = load(os.path.join(SCENE, "t_night_v1.png"), gray=True)
    x0, y0, x1, y1 = person_box(t)
    ph, cx, cy = y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2
    ch = int(1.5 * ph); cw = int(ch * 300 / 64)
    ox, oy = int(cx - cw * 0.42), int(cy - ch / 2)
    crop = palette(t)[oy:oy + ch, ox:ox + cw]
    k = 192 / ch
    bt = np.array([x0 - 6, y0 - 6, x1 + 6, y1 + 6], float)                 # T detector: tight, score 0.82
    br = bt + np.array([0.45, -0.06, 0.60, 0.10]) * ph                    # RGB detector at night: shifted, loose, 0.31
    st, sr = 0.82, 0.31
    wbf = (st * bt + sr * br) / (st + sr)

    def tile(boxes, text):
        im = to_img(crop).resize((int(cw * k), 192), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        for b, c in boxes:
            d.rectangle([(b[0] - ox) * k, (b[1] - oy) * k, (b[2] - ox) * k, (b[3] - oy) * k], outline=c, width=5)
        tag(im, (14, 12), text, 42)
        return im

    arrow = Image.new("RGB", (168, 192), CARD)
    ImageDraw.Draw(arrow).text((48, 50), "→", font=ImageFont.truetype(FONT, 80), fill=MUTED)
    both = tile([(br, BLUE), (bt, ORANGE)], "RGB 0.31 · T 0.82")
    nms = row([both, arrow, tile([(bt, ORANGE)], "NMS → 0.82")], 0)
    wb = row([both, arrow, tile([(wbf, GREEN)], f"WBF → {(st + sr) / 2:.2f}")], 0)
    return save(nms, "p33_nms.jpg"), save(wb, "p33_wbf.jpg")


# ---------- slide 38 and appendix platforms ----------
def p38():
    rgb, t = load(os.path.join(GEN, "p38_rgb_v1.png")), load(os.path.join(GEN, "p38_t_v1.png"), gray=True)
    x0, y0, x1, y1, cut = 560, 430, 2000, 1330, 1290
    a = np.concatenate([night(rgb)[y0:y1, x0:cut], palette(t)[y0:y1, cut:x1]], axis=1)
    im = to_img(a)
    d = ImageDraw.Draw(im)
    d.line([(cut - x0, 0), (cut - x0, y1 - y0)], fill=(230, 232, 235), width=5)
    s = (y1 - y0) // 14
    tag(im, (s, s), "RGB", s); tag(im, (cut - x0 + s, s), "LWIR", s)
    sub = t[y0:y1, cut:x1]                                           # the walker: brightest small blob on the road
    lab, n = ndimage.label(sub > np.percentile(sub, 99.9))
    sizes = ndimage.sum(np.ones_like(sub), lab, range(1, n + 1))
    cands = [i + 1 for i in range(n) if 15 < sizes[i] < 900]
    if cands:
        cyx = [ndimage.center_of_mass(lab == i) for i in cands]
        py, px = min(cyx, key=lambda c: -c[0])                       # lowest in the frame = nearest on the road
        px += cut - x0
        d.ellipse([px - 46, py - 46, px + 46, py + 46], outline=(255, 255, 255), width=4)
        tag(im, (int(px) + 60, int(py) - 20), "людина", s)
    return save(im, "p38_isr.jpg")


def platform(name, out):
    im = Image.open(os.path.join(GEN, name + ".png")).convert("RGB")
    W, H = im.size
    w = int(H * 1.6)
    return save(im.crop(((W - w) // 2, 0, (W - w) // 2 + w, H)).resize((1248, 780), Image.LANCZOS), out)


# ---------- memes ----------
MEMES = [  # name, render, top, bottom, demo slide
    ("m1_crossover", "m1_crossover_v1", "Тепловізор бачить усе", "14:00, асфальт +35 °C", 12),
    ("m2_agc", "m2_agc_v1", "AGC побачив одну гарячу точку", "Решта кадру: на все добре", 18),
    ("m3_rms", "m3_rms_v1", "RMS 0.29 px", "Стіна на 75 м: 17 px убік", 26),
    ("m4_zeros", "m4_zeros_v1", "T-канал з нульовими вагами", "Я тут для shape", 31),
    ("m5_shift", "m5_shift_v1", "Early fusion: зсув 4 px", "Це одна людина чи дві?", 36),
    ("b1_kettle", "b1_kettle_v1", "Тепловізор: -22 °C", "Чайник: я дзеркало", None),
    ("b2_ffc", "b2_ffc_v1", "FFC: клац", "Калман: ціль зупинилась!", None),
]


def classic(im, top, bottom):
    """White Impact caption with a black outline, top and bottom."""
    d = ImageDraw.Draw(im)
    W, H = im.size
    for text, at_top in ((top.upper(), True), (bottom.upper(), False)):
        size = int(0.105 * H)
        while size > 10 and d.textlength(text, font=ImageFont.truetype(IMPACT, size)) > 0.94 * W:
            size -= 2
        f = ImageFont.truetype(IMPACT, size)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f, stroke_width=4)
        y = 0.035 * H if at_top else H - 0.035 * H - (y1 - y0)
        d.text(((W - (x1 - x0)) / 2 - x0, y - y0), text, font=f, fill="white", stroke_width=max(3, size // 14),
               stroke_fill="black")
    return im


def memes():
    made = {}
    for name, render, top, bottom, slide in MEMES:
        im = Image.open(os.path.join(MEME_SRC, render + ".png")).convert("RGB").resize((1200, 896), Image.LANCZOS)
        im = classic(im, top, bottom)
        im.save(os.path.join(MEME_OUT, name + ".jpg"), quality=92)
        made[name] = (im, slide)
    print("memes", len(made))
    return made


# ---------- mock-ups on 1600x900 thumbnails ----------
def mock(thumbs, out, slide, items, name=None, wipe=(), texts=()):
    """items: (image, x, y, w); wipe: (x, y, w, h) regions repainted with the colour at their top-left corner + 2 px."""
    bg = Image.open(os.path.join(thumbs, f"s{slide:02d}.png")).convert("RGB")
    d = ImageDraw.Draw(bg)
    for x, y, w, h, c in wipe:
        d.rectangle((x, y, x + w, y + h), fill=c)
    for im, x, y, w in items:
        bg.paste(im.resize((w, round(im.height * w / im.width)), Image.LANCZOS), (x, y))
    for x, y, text, size, fill in texts:
        d.text((x, y), text, font=ImageFont.truetype(FONT, size), fill=fill)
    bg.save(os.path.join(out, name or f"mock_s{slide:02d}.png"))
    print("mock", slide)


if __name__ == "__main__":
    pair, drone, chain, levels, ffc, gimbal = p8(), p11(), p14(), p17(), p16(), p20()
    nms, wbf = p33()
    isr = p38()
    plats = [platform("p55_fpv_v1", "p55_fpv.jpg"), platform("p56_mast_v1", "p56_mast.jpg"),
             platform("p57_ugv_v1", "p57_ugv.jpg")]
    mm = memes()
    if len(sys.argv) > 3:
        th, mo = sys.argv[2], sys.argv[3]
        mock(th, mo, 8, [(pair, 792, 288, 728)], wipe=[(790, 286, 732, 430, SLIDE)],
             texts=[(792, 660, "Та сама сцена: людину за склом LWIR не бачить, полірований лист «показує» холодне небо",
                     15, MUTED)])
        mock(th, mo, 11, [(drone, 106, 520, 688)])
        mock(th, mo, 14, [(chain[0], 104, 544, 416), (chain[1], 592, 544, 416), (chain[2], 1080, 544, 416)])
        mock(th, mo, 16, [(ffc, 80, 568, 360)],
             texts=[(470, 586, "ліворуч: шторка відкрита,", 17, MUTED), (470, 610, "матриця бачить сцену", 17, MUTED),
                    (470, 648, "праворуч: FFC — шторка закрита,", 17, ORANGE), (470, 672, "кадр «замерзає»", 17, ORANGE)])
        mock(th, mo, 17, [(levels, 106, 524, 688)])
        mock(th, mo, 20, [(gimbal, 106, 520, 690)])
        mock(th, mo, 33, [(nms, 104, 338, 656), (wbf, 840, 338, 656)])
        mock(th, mo, 38, [(isr, 104, 426, 416)])
        for s, im in zip((55, 56, 57), plats):
            mock(th, mo, s, [(im, 104, 426, 416)])
        for name, (im, slide) in mm.items():
            if slide:
                mock(th, mo, slide, [(im, 1150, 425, 370)])
