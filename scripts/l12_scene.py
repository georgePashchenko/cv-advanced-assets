"""Lesson 12: slide images from one Gemini scene (road, walker ~8 m, van, house, poles, metal sheet).

Usage: l12_scene.py <versions_dir> <out_dir> <slide> [...]
The renders in versions/ share one geometry: base_rgb (overcast day) and Gemini edits of it, i.e. rgb_night,
rgb_smoke (v2), t_night, t_cross, t_smoke (LWIR, white-hot). Prompts are next to them. Effects that carry
numbers on the slides (blur, AGC, FPN, edge shifts) are applied here, not by Gemini.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
FONT = "C:/Windows/Fonts/seguisb.ttf"
CARD = (30, 33, 36)
rng = np.random.default_rng(12)


def load(name, gray=False):
    im = Image.open(os.path.join(src, name + ".png"))
    return np.asarray(im.convert("L" if gray else "RGB")).astype(float) / 255


def night(rgb):
    """Visible camera at 23:00 without lighting: little signal, high gain -> dark and grainy."""
    g = 0.22 * rgb ** 1.6 * np.array([0.85, 1.0, 0.9])
    g += rng.normal(0, 0.035, g.shape[:2])[..., None] + rng.normal(0, 0.015, g.shape)
    return g.clip(0, 1)


def stretch(t, lo=1, hi=99.7, gamma=1.4):
    a, b = np.percentile(t, [lo, hi])
    return np.clip((t - a) / (b - a), 0, 1) ** gamma


def palette(t, cmap="inferno"):
    return plt.get_cmap(cmap)(stretch(t))[..., :3]


def tag(im, xy, text, size):
    d = ImageDraw.Draw(im, "RGBA")
    f = ImageFont.truetype(FONT, size)
    x0, y0, x1, y1 = d.textbbox(xy, text, font=f)
    p = size // 3
    d.rounded_rectangle((x0 - p, y0 - p, x1 + p, y1 + p), radius=p, fill=(0, 0, 0, 150))
    d.text(xy, text, font=f, fill=(232, 236, 234))


def to_img(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def split(left, right, box, cut):
    """Crop box (x0, y0, x1, y1) of two aligned renders, left part from `left`, right part from `right`."""
    x0, y0, x1, y1 = box
    a = np.concatenate([left[y0:y1, x0:cut], right[y0:y1, cut:x1]], axis=1)
    im = to_img(a)
    d = ImageDraw.Draw(im)
    d.line([(cut - x0, 0), (cut - x0, y1 - y0)], fill=(230, 232, 235), width=4)
    s = (y1 - y0) // 13
    tag(im, (s, s), "RGB", s)
    tag(im, (cut - x0 + s, s), "LWIR", s)
    return im


def p4():
    base, smoke = load("base_rgb_v1"), load("rgb_smoke_v2")
    tn, tc, ts = (load(n, gray=True) for n in ("t_night_v1", "t_cross_v1", "t_smoke_v1"))
    box, cut = (150, 330, 2290, 1320), 1150  # aspect 2.16 = card image; van left of the cut, walker right
    for name, l, r in (("night", night(base), palette(tn)), ("cross", base, palette(tc)),
                       ("smoke", smoke, palette(ts))):
        split(l, r, box, cut).save(os.path.join(out, f"p4_{name}.jpg"), quality=92)


def crop_aspect(a, aspect, cx, cy, w):
    """Crop of width w (or less if it does not fit) and given aspect, centred near (cx, cy) inside the image."""
    H, W = a.shape[:2]
    w = min(w, W, int(H * aspect)); h = int(w / aspect)
    x0 = int(np.clip(cx - w / 2, 0, W - w)); y0 = int(np.clip(cy - h / 2, 0, H - h))
    return a[y0:y0 + h, x0:x0 + w]


def resize(a, w, h=None):
    im = to_img(a)
    h = h or round(im.height * w / im.width)
    return np.asarray(im.resize((w, h), Image.LANCZOS)).astype(float) / 255


def sensor(t, w=640):
    """LWIR frame at microbolometer resolution."""
    return resize(t, w)


def person_box(t):
    """Bounding box of the walker = largest very bright blob of the night LWIR frame."""
    from scipy import ndimage
    lab, n = ndimage.label(t > np.percentile(t, 99.3))
    k = np.argmax(ndimage.sum(np.ones_like(t), lab, range(1, n + 1))) + 1
    ys, xs = np.nonzero(lab == k)
    return xs.min(), ys.min(), xs.max(), ys.max()


def panels(imgs, labels, gap, size, colors=None, bg=CARD):
    """Panels side by side with a small label above each, on the card colour."""
    h, w = imgs[0].shape[:2]
    lab_h = int(size * 1.9)
    canvas = Image.new("RGB", (len(imgs) * w + (len(imgs) - 1) * gap, h + lab_h), bg)
    d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT, size)
    for i, (a, s) in enumerate(zip(imgs, labels)):
        x = i * (w + gap)
        canvas.paste(to_img(a if a.ndim == 3 else np.repeat(a[..., None], 3, 2)), (x, lab_h))
        d.text((x, int(size * 0.3)), s, font=f, fill=(colors[i] if colors else (160, 166, 163)))
    return canvas


def save(im, name, q=92):
    (im if isinstance(im, Image.Image) else to_img(im)).save(os.path.join(out, name), quality=q)


def p5():
    """RGB at night and a radiometric-style frame with a °C colour bar (aspect 2.077 each)."""
    base, tn = load("base_rgb_v1"), load("t_night_v1", gray=True)
    asp = 13500 / 6500
    save(crop_aspect(night(base), asp, 1376, 800, 2752), "p5_rgb.jpg")
    # white-hot grey level -> illustrative temperature: sky about -25 °C, walker about +32 °C
    a, b = np.percentile(tn, [0.5, 99.9])
    T = -25 + 57 * np.clip((tn - a) / (b - a), 0, 1)
    crop = crop_aspect(T, asp * 0.8, 1376, 800, 2752)
    h, w = crop.shape
    fig = plt.figure(figsize=(asp * 4, 4), dpi=h / 4)
    ax = fig.add_axes([0, 0, 0.8, 1]); ax.axis("off")
    im = ax.imshow(crop, cmap="inferno", vmin=-25, vmax=32, aspect="auto")
    cax = fig.add_axes([0.83, 0.06, 0.03, 0.72])
    cb = fig.colorbar(im, cax=cax, ticks=[-20, 0, 20])
    fs = 22  # pt at 4 in figure height: ~8 pt on the 1.2 in tall slide image
    cb.ax.tick_params(colors="#c3c7cd", labelsize=fs, length=0)
    cb.ax.set_yticklabels(["-20", "0", "20"])
    cb.outline.set_visible(False)
    fig.text(0.845, 0.86, "°C", color="#c3c7cd", fontsize=fs, ha="center")
    fig.patch.set_facecolor(np.array(CARD) / 255)
    fig.savefig(os.path.join(out, "p5_radiometric.png"), facecolor=fig.get_facecolor())


def divider(name, a, aspect=23350 / 17500, cx=1250, cy=860, w=1900):
    save(crop_aspect(a, aspect, cx, cy, w), name)


def fpn(t, k=1.0, seed=5):
    """Uncorrected microbolometer: column offsets, a few row stripes, pixel gain spread, dead pixels."""
    r = np.random.default_rng(seed)
    H, W = t.shape
    col = r.normal(0, 0.022, W)[None, :]
    row = np.zeros((H, 1)); row[r.choice(H, 6, replace=False)] = r.normal(0, 0.05, (6, 1))
    gain = 1 + r.normal(0, 0.03, (H, W))
    f = t * (1 + k * (gain - 1)) + k * (col + row) + r.normal(0, 0.012, (H, W))
    if k == 1:
        dead = r.choice(H * W, 25, replace=False)
        f.flat[dead] = r.choice([0.0, 1.0], 25)
    return f.clip(0, 1)


def p6():
    divider("p6_lwir.jpg", palette(load("t_night_v1", gray=True)))


def p14():
    t = sensor(load("t_night_v1", gray=True), 960)
    divider("p14_raw.jpg", fpn(t), cx=450, cy=310, w=700)


def p9():
    t = load("t_night_v1", gray=True)
    save(crop_aspect(t, 22000 * 189.1145 / (12000 * 182.88), 1376, 820, 2752), "p9_lwir.jpg")


def p11():
    """Thermal smear: microbolometer as a first-order lag, trail = exp(-x / (ω·τ / IFOV)) px."""
    t = sensor(load("t_night_v1", gray=True))
    x0, y0, x1, y1 = person_box(t)
    cx, cy, hh = (x0 + x1) // 2, (y0 + y1) // 2, int(0.8 * (y1 - y0))
    ww = int(hh * 520 / 340)
    out_imgs = []
    for L in (0.0, 4.3, 12.8):
        if L:
            n = int(6 * L) + 1
            k = np.exp(-np.arange(n) / L); k /= k.sum()
            b = np.apply_along_axis(lambda r: np.convolve(r, k)[:len(r)], 1, t)  # gimbal pans right
        else:
            b = t
        c = b[cy - hh:cy + hh, cx - int(0.4 * ww):cx + int(0.6 * ww)]  # walker left of centre, room for the trail
        out_imgs.append(resize(c, 520, 340))
    save(panels(out_imgs, ["0 °/с · 0.0 px", "30 °/с · 4.3 px", "90 °/с · 12.8 px"], 26, 40), "p11_smear.jpg")


def agc(c, mode, lo=None, hi=None):
    if mode == "minmax":
        return (c - c.min()) / (c.max() - c.min())
    if mode == "pct":
        a, b = np.percentile(c, [0.5, 99.9])
        return np.clip((c - a) / (b - a), 0, 1)
    v = np.sort(c.ravel())
    return np.searchsorted(v, c, side="right") / v.size


def p16():
    """AGC on a raw-like frame without and with a small 600 K source (< 0.1 % of pixels)."""
    t = sensor(load("t_night_v1", gray=True))
    hot = t.copy()
    H, W = t.shape
    yy, xx = np.mgrid[:H, :W]
    hot[(xx - 175) ** 2 + (yy - 128) ** 2 < 36] = 9.0  # burning barrel by the van: far above any 300 K level
    imgs, labels = [], []
    for frame, tagtxt in ((t, ""), (hot, "+ 600 K: ")):
        for mode, name in (("minmax", "min–max"), ("pct", "перцентиль"), ("eq", "histeq")):
            imgs.append(resize(crop_aspect(agc(frame, mode), 1.45, 330, 180, 480), 520, 358))
            labels.append(tagtxt + name)
    top = panels(imgs[:3], labels[:3], 22, 34)
    bot = panels(imgs[3:], labels[3:], 22, 34, colors=[(240, 122, 30), (160, 166, 163), (160, 166, 163)])
    canvas = Image.new("RGB", (top.width, top.height + bot.height + 18), CARD)
    canvas.paste(top, (0, 0)); canvas.paste(bot, (0, top.height + 18))
    save(canvas, "p16_agc.jpg")


def p18():
    t = sensor(load("t_night_v1", gray=True))
    a, b = fpn(t), fpn(t, k=0.05)
    cut = lambda f: resize(crop_aspect(f, 600 / 437, 330, 175, 460), 600, 437)  # overall 2.44 = slide frame
    save(panels([cut(a), cut(b)], ["до NUC: FPN і смуги", "після NUC: залишок 5 %"], 24, 34), "p18_nuc.jpg")


F_RGB, B = 1108.0, 0.051  # RGB grid 1280 px wide, baseline 5.1 cm (Demo 2)


def disparity_field(t1280, z0):
    """Residual shift f·B·(1/Z - 1/Z0) px of a plane homography fitted at Z0, per pixel of the 1280-wide RGB grid.
    Depth: walker 8 m (thermal blob), ground plane below the horizon (camera height from the walker's feet),
    everything above the horizon ~40 m."""
    H, W = t1280.shape
    x0, y0, x1, y1 = person_box(t1280)
    horizon = 0.262 * H
    h_cam = 8.0 * (y1 - horizon) / F_RGB
    rows = np.arange(H)[:, None].astype(float)
    Z = np.where(rows > horizon + 1, h_cam * F_RGB / np.maximum(rows - horizon, 1), 40.0) * np.ones((1, W))
    Z = np.minimum(Z, 40.0)
    Z[y0:y1 + 1, x0:x1 + 1][t1280[y0:y1 + 1, x0:x1 + 1] > np.percentile(t1280, 97)] = 8.0
    inv0 = 0 if z0 is None else 1 / z0
    return F_RGB * B * (1 / Z - inv0)


def edges(t, q=96):
    from scipy import ndimage
    g = np.hypot(ndimage.sobel(t, 0), ndimage.sobel(t, 1))
    return g > np.percentile(g, q)


def p25():
    """T edges (red) on RGB around the walker: H∞, plane at 18 m, per-pixel depth."""
    rgb = resize(load("base_rgb_v1"), 1280)
    t = resize(load("t_night_v1", gray=True), 1280)
    e = edges(t)
    x0, y0, x1, y1 = person_box(t)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    imgs, labels = [], []
    for z0, lab in ((None, "H∞"), (18.0, "H(Z0 = 18 м)"), ("depth", "з глибиною")):
        d = np.zeros_like(t) if z0 == "depth" else disparity_field(t, z0)
        ys, xs = np.nonzero(e)
        xs2 = np.clip(np.round(xs + d[ys, xs]).astype(int), 0, t.shape[1] - 1)
        im = rgb.copy() * 0.8
        im[ys, xs2] = [0.94, 0.27, 0.2]
        shift = 0 if z0 == "depth" else F_RGB * B * (1 / 8 - (0 if z0 is None else 1 / z0))
        imgs.append(resize(im[cy - 120:cy + 130, cx - 83:cx + 84], 367, 550))  # overall 1.93 = slide frame
        labels.append(f"{lab}: {shift:.1f} px")
    save(panels(imgs, labels, 18, 30), "p25_edges.jpg")


def p22():
    """RGB with the LWIR frame warped by H∞ on top: everything near is doubled."""
    rgb = resize(load("base_rgb_v1"), 1280)
    t = resize(load("t_night_v1", gray=True), 1280)
    d = disparity_field(t, None)
    H, W = t.shape
    xs = np.clip(np.arange(W)[None, :] - d, 0, W - 1).astype(int)
    tw = t[np.arange(H)[:, None], xs]
    a = 0.5 * rgb + 0.5 * palette(tw)
    divider("p22_overlay.jpg", a, cx=640, cy=400, w=900)


def p33():
    """Conditions x modality: day, night, crossover, smoke; RGB top, LWIR bottom (4:3 overall)."""
    base, smoke = load("base_rgb_v1"), load("rgb_smoke_v2")
    tn, tc, ts = (load(n, gray=True) for n in ("t_night_v1", "t_cross_v1", "t_smoke_v1"))
    cols = [(base, ts, "day"), (night(base), tn, "night"), (base, tc, "crossover"), (smoke, ts, "smoke")]
    tiles = [[resize(crop_aspect(c, 0.95, 1150, 820, 1100), 400, 421) for c in (r, palette(t))] for r, t, _ in cols]
    rgb_row = panels([x[0] for x in tiles], [c[2] for c in cols], 14, 30)
    t_row = panels([x[1] for x in tiles], ["", "", "", ""], 14, 30)
    t_row = t_row.crop((0, int(30 * 1.9) - 14, t_row.width, t_row.height))
    canvas = Image.new("RGB", (rgb_row.width, rgb_row.height + t_row.height), CARD)
    canvas.paste(rgb_row, (0, 0)); canvas.paste(t_row, (0, rgb_row.height))
    save(canvas, "p33_grid.jpg")


def p45():
    """2 x 2: RGB | LWIR at night and in smoke."""
    base, smoke = load("base_rgb_v1"), load("rgb_smoke_v2")
    tn, ts = load("t_night_v1", gray=True), load("t_smoke_v1", gray=True)
    tl = lambda a: resize(crop_aspect(a, 1.334, 1250, 820, 1500), 640, 480)
    top = np.concatenate([tl(night(base)), tl(palette(tn))], 1)
    bot = np.concatenate([tl(smoke), tl(palette(ts))], 1)
    save(np.concatenate([top, bot], 0), "p45_grid.jpg")


def grid_centres(g, dark, rows=5, cols=7):
    """Centroids of the 35 discs (threshold + connected components), ordered row by row."""
    from scipy import ndimage
    m = g < 0.3 if dark else g > 0.75
    if dark:  # only inside the white board, not the tripod or the grass
        lab, n = ndimage.label(g > 0.7)
        board = lab == np.argmax(ndimage.sum(np.ones_like(g), lab, range(1, n + 1))) + 1
        m &= ndimage.binary_fill_holes(board)
    lab, n = ndimage.label(ndimage.binary_opening(m, iterations=3))
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    keep = np.argsort(sizes)[::-1][:rows * cols] + 1
    c = np.array(ndimage.center_of_mass(m, lab, keep))  # (y, x)
    c = c[np.argsort(c[:, 0])]
    return [r[np.argsort(r[:, 1])] for r in np.split(c, rows)]


def p27():
    """Calibration target (separate Gemini pair, slide27/versions): RGB dark discs, LWIR hot discs, found centres."""
    d = os.path.join(os.path.dirname(src.rstrip("/\\")), "..", "slide27", "versions")
    ims = []
    colours = [(240, 80, 60), (242, 161, 42), (46, 196, 182), (56, 189, 248), (192, 132, 252)]
    for name, dark in (("target_rgb_v1", True), ("target_t_v1", False)):
        im = Image.open(os.path.join(d, name + ".png")).convert("RGB")
        g = np.asarray(im.convert("L")).astype(float) / 255
        rows = grid_centres(g, dark)
        dr = ImageDraw.Draw(im)
        for i, r in enumerate(rows):
            pts = [(x, y) for y, x in r]
            dr.line(pts, fill=colours[i], width=6)
            for x, y in pts:
                dr.ellipse((x - 16, y - 16, x + 16, y + 16), outline=colours[i], width=6)
        a = np.asarray(im).astype(float) / 255
        ims.append(resize(crop_aspect(a, 720 / 523, 1200, 860, 2400), 720, 523))  # overall 2.49 = slide frame
    save(panels(ims, ["RGB: темні кола", "LWIR: гарячі кола"], 24, 34), "p27_target.jpg")


for s in sys.argv[3:]:
    globals()[s]()
    print("done", s)
