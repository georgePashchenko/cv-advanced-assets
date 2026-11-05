"""Edit an image with Gemini: gemini_edit.py <image> <prompt.txt> <out> [aspect] [model].

Pass "-" as <image> to generate from the prompt alone.

Reads the API key from GEMINI_API_KEY.
"""
import base64
import json
import mimetypes
import os
import sys
import urllib.request

img_path, prompt_path, out_path = sys.argv[1:4]
aspect = sys.argv[4] if len(sys.argv) > 4 else None
model = sys.argv[5] if len(sys.argv) > 5 else "gemini-3-pro-image"

parts = []
if img_path != "-":
    with open(img_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    parts.append({"inlineData": {"mimeType": mimetypes.guess_type(img_path)[0] or "image/png", "data": data}})
with open(prompt_path, encoding="utf-8") as f:
    prompt = f.read()

size = os.environ.get("GEMINI_IMAGE_SIZE")  # "1K" | "2K" | "4K"
gen = {"responseModalities": ["IMAGE"], "imageConfig": {}}
if aspect:
    gen["imageConfig"]["aspectRatio"] = aspect
if size:
    gen["imageConfig"]["imageSize"] = size
body = {
    "contents": [{"parts": parts + [{"text": prompt}]}],
    "generationConfig": gen,
}
req = urllib.request.Request(
    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
)
with urllib.request.urlopen(req, timeout=300) as r:
    resp = json.load(r)

for part in resp["candidates"][0]["content"]["parts"]:
    if "inlineData" in part:
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(part["inlineData"]["data"]))
        print("saved", out_path, part["inlineData"]["mimeType"])
        break
else:
    print(json.dumps(resp, ensure_ascii=False)[:1500])
