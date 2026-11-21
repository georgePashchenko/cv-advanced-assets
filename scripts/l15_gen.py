"""Lesson 15: slide images and captioned memes from Gemini renders (no text in the renders).

Usage: l15_gen.py <repo_root>
Reads lesson15/gen/versions and lesson15/memes/versions, writes lesson15/gen/*.jpg and lesson15/memes/*.jpg.
"""
import os
import sys

from PIL import Image

ROOT = sys.argv[1]
sys.argv[1] = os.path.join(ROOT, "lesson12")           # l12_gen2 reads its lesson dir from argv on import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l12_gen2 import classic                           # noqa: E402

GEN, OUT = os.path.join(ROOT, "lesson15/gen/versions"), os.path.join(ROOT, "lesson15/gen")
MEME_SRC, MEME_OUT = os.path.join(ROOT, "lesson15/memes/versions"), os.path.join(ROOT, "lesson15/memes")

# name, output, aspect of the free space on the slide, vertical centre of the band (fraction of height)
BANDS = [("i2_server", "p2_server.jpg", 656 / 190, 0.45), ("i2_onboard", "p2_onboard.jpg", 656 / 190, 0.50),
         ("i15_leaves", "p15_leaves.jpg", 16 / 9, 0.5), ("i15_branches", "p15_branches.jpg", 16 / 9, 0.5),
         ("i28_ground", "p28_ground.jpg", 656 / 146, 0.40), ("i28_air", "p28_air.jpg", 656 / 146, 0.47)]

MEMES = [("m10_stopwatch", "Замір без прогріву", "Похибка ×21"),
         ("m11_tiny_car", "Найменше FLOPs", "Сьоме місце на CPU"),
         ("m16_bus", "Обнулили 80 % ваг", "Автобус той самий, пробка та сама"),
         ("m19_hedge", "Прунінг: -30 % каналів", "train(): перебудував з yaml"),
         ("m26_books", "Teacher: ось усі ознаки neck", "Student: було 0.137, стало 0.129"),
         ("m37_laptop", "Працює на моєму ноутбуці", "Борт: 77 мс на кадр")]


def band(im, aspect, centre):
    """Full-width crop of the given aspect around a vertical centre."""
    W, H = im.size
    h = min(H, int(W / aspect))
    y = int(min(max(centre * H - h / 2, 0), H - h))
    return im.crop((0, y, W, y + h))


if __name__ == "__main__":
    for name, out, aspect, centre in BANDS:
        im = band(Image.open(os.path.join(GEN, name + "_v1.png")).convert("RGB"), aspect, centre)
        im.thumbnail((2000, 2000))
        im.save(os.path.join(OUT, out), quality=90)
        print("saved", out, im.size)
    for name, top, bottom in MEMES:
        im = Image.open(os.path.join(MEME_SRC, name + "_v1.png")).convert("RGB").resize((1200, 896), Image.LANCZOS)
        classic(im, top, bottom).save(os.path.join(MEME_OUT, name.split("_")[0] + ".jpg"), quality=92)
    print("memes", len(MEMES))
