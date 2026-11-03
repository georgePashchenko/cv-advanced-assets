"""Overlay captions onto the user's two-panel meme template: meme_text.py <template> <out>."""
import sys
from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
img = Image.open(src).convert("RGB")
d = ImageDraw.Draw(img)

# White caption cells of the 960x720 template (x0, y0, x1, y1)
cells = [
    ((374, 6, 954, 350), "«Max range\n200 м»"),
    ((374, 366, 954, 712), "«40 м @ 10 %\nвідбивної\nздатності»"),
]
for (x0, y0, x1, y1), text in cells:
    d.rectangle((x0, y0, x1, y1), fill="white")  # wipe placeholder text
    for size in range(64, 20, -2):
        font = ImageFont.truetype(r"C:\Windows\Fonts\segoeuib.ttf", size)
        l, t, r, b = d.multiline_textbbox((0, 0), text, font=font, align="center", spacing=10)
        if r - l <= (x1 - x0) * 0.86 and b - t <= (y1 - y0) * 0.8:
            break
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    d.multiline_text((cx - (r - l) / 2 - l, cy - (b - t) / 2 - t), text, font=font,
                     fill="black", align="center", spacing=10)

img.save(out, quality=95)
print(img.size, out)
