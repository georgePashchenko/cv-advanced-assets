"""Two-buttons meme for slide 20 on the user's template: meme_two_buttons.py <template> <out>.

Plate geometry is measured on the 449x683 template; everything is drawn on a 3x upscale.
"""
import sys

from PIL import Image, ImageDraw, ImageFont

src, out = sys.argv[1], sys.argv[2]
K = 3
tpl = Image.open(src).convert("RGB")
tpl = tpl.resize((tpl.width * K, tpl.height * K), Image.LANCZOS)

MONO = r"C:\Windows\Fonts\consolab.ttf"
BOLD = r"C:\Windows\Fonts\segoeuib.ttf"

# (text, centre on the plate above the button, plate top-edge angle in degrees CCW)
LABELS = [("T", (120, 94), 10.2), ("inv(T)", (268, 80), 6.8)]
font = ImageFont.truetype(MONO, 86)
for text, (cx, cy), ang in LABELS:
    l, t, r, b = font.getbbox(text)
    layer = Image.new("RGBA", (r - l + 40, b - t + 40), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((20 - l, 20 - t), text, font=font, fill=(20, 20, 24, 255))
    layer = layer.rotate(ang, resample=Image.BICUBIC, expand=True)
    tpl.paste(layer, (cx * K - layer.width // 2, cy * K - layer.height // 2), layer)

# Caption strip under the comic
cap_h = 330
canvas = Image.new("RGB", (tpl.width, tpl.height + cap_h), "white")
canvas.paste(tpl, (0, 0))
d = ImageDraw.Draw(canvas)
cap = "Tr_velo_to_cam —\nце «з чого в що»?"
cfont = ImageFont.truetype(BOLD, 96)
l, t, r, b = d.multiline_textbbox((0, 0), cap, font=cfont, align="center", spacing=14)
d.multiline_text(((canvas.width - (r - l)) / 2 - l, tpl.height + (cap_h - (b - t)) / 2 - t - 10), cap,
                 font=cfont, fill="black", align="center", spacing=14)
canvas.save(out, quality=95)
print(canvas.size, f"aspect {canvas.width / canvas.height:.3f}")
