import json, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(Path(__file__).parent))
from detectors import nms
ROOT = Path(__file__).resolve().parent.parent; SRC = Path("/workspace/rag_civitai")
raw = json.load(open(ROOT / "data/bench_raw.json")); sample = json.load(open(ROOT / "data/bench_sample.json"))
font = ImageFont.load_default(size=22)
union = {}
for r in sample:
    k = str(r["id"]); allb = []
    for name, d in raw.items():
        thr = 0.25 if name == "gdino_tiny" else 0.15
        allb += [tuple(b) for b in d["boxes"][k] if b[4] >= thr]
    union[k] = nms(allb, 0.4)
json.dump(union, open(ROOT / "data/bench_union.json", "w"))
tiles = []
for n, r in enumerate(sample):
    im = Image.open(SRC / r["path"]).convert("RGB"); s = min(380 / im.width, 540 / im.height)
    im = im.resize((int(im.width * s), int(im.height * s))); d = ImageDraw.Draw(im)
    for j, b in enumerate(union[str(r["id"])]):
        bb = [v * s for v in b[:4]]; d.rectangle(bb, outline=(255, 0, 0), width=3)
        d.text((bb[0] + 2, bb[1] + 2), str(j), fill=(255, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((4, 4), f"#{n} {r['style']}", fill=(0, 255, 0), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    tiles.append(im)
for g in range(4):
    T = tiles[g * 12:(g + 1) * 12]; H = 540
    canvas = Image.new("RGB", (380 * 4, H * 3), "white")
    for i, t in enumerate(T):
        pass
        canvas.paste(t, ((i % 4) * 380, (i // 4) * H))
    canvas.save(ROOT / f"qa/bench_union_{g}.jpg", quality=85)
