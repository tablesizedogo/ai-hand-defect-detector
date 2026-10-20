"""Stage 4: visual QA grids of the pre-annotations (per style) + crop sheets sorted by the CLIP defect hint."""
import json, random, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parent.parent; SRC = Path("/workspace/rag_civitai")
rows = [json.loads(l) for l in open(ROOT / "data/preannotations.jsonl")]
font = ImageFont.load_default(size=18)

def tile(r, tw=320, th=440, hint_colors=True):
    im = Image.open(SRC / r["path"]).convert("RGB"); s = min(tw / im.width, th / im.height)
    im = im.resize((int(im.width * s), int(im.height * s))); d = ImageDraw.Draw(im)
    for b in r["boxes"]:
        x0, y0, x1, y1 = [v * s for v in b["xyxy"]]; h = b.get("defect_hint", 0)
        col = ((239, 68, 68) if h > 0.6 else (250, 204, 21) if h > 0.4 else (34, 197, 94)) if hint_colors else (0, 229, 255)
        d.rectangle([x0, y0, x1, y1], outline=col, width=3)
        d.text((x0 + 2, y0 + 1), f"{b['score']:.2f}", fill=col, font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((3, 3), f"{r['family']}/{r['style']} {r['id']}", fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return im

def grid(items, cols, tw, th, out):
    rws = (len(items) + cols - 1) // cols
    c = Image.new("RGB", (cols * tw, rws * th), (30, 30, 30))
    for i, t in enumerate(items):
        c.paste(t, ((i % cols) * tw + (tw - t.width) // 2, (i // cols) * th + (th - t.height) // 2))
    c.save(out, quality=85)

if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    random.seed(seed)
    for st in ["photo", "anime", "art"]:
        pick = random.sample([r for r in rows if r["style"] == st], 12)
        grid([tile(r) for r in pick], 4, 320, 440, ROOT / f"qa/qa_{st}_s{seed}.jpg")
        json.dump([r["id"] for r in pick], open(ROOT / f"qa/qa_{st}_s{seed}_ids.json", "w"))
    # crop sheets by hint (for picking guide examples)
    crops = []
    for r in rows:
        for j, b in enumerate(r["boxes"]):
            x0, y0, x1, y1 = b["xyxy"]
            if min(x1 - x0, y1 - y0) >= 48: crops.append((b.get("defect_hint", 0), r, j))
    crops.sort(key=lambda c: -c[0])
    def sheet(sel, out):
        ts = []
        for k, (h, r, j) in enumerate(sel):
            im = Image.open(SRC / r["path"]).convert("RGB"); x0, y0, x1, y1 = r["boxes"][j]["xyxy"]
            p = 0.15 * max(x1 - x0, y1 - y0); c = im.crop((max(0, x0 - p), max(0, y0 - p), min(im.width, x1 + p), min(im.height, y1 + p)))
            c.thumbnail((200, 200)); t = Image.new("RGB", (200, 220), (30, 30, 30)); t.paste(c, ((200 - c.width) // 2, 20))
            ImageDraw.Draw(t).text((3, 1), f"{k} h={h:.2f} {r['style'][:2]}", fill=(255, 255, 255), font=font)
            ts.append(t)
        grid(ts, 8, 200, 220, out)
        json.dump([[r["id"], j, h] for h, r, j in sel], open(str(out).replace(".jpg", ".json"), "w"))
    sheet(crops[:48], ROOT / "qa/crops_top_hint.jpg")
    sheet(crops[48:96], ROOT / "qa/crops_top_hint_2.jpg")
    random.seed(1); low = [c for c in crops if c[0] < 0.15]
    sheet(random.sample(low, 48), ROOT / "qa/crops_low_hint.jpg")
    print("crops >=48px:", len(crops), "hint>0.6:", sum(c[0] > 0.6 for c in crops), "hint<0.15:", len(low))
