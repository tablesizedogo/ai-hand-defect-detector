"""Build docs/examples_ok_bad.jpg and docs/examples_edge_cases.jpg from hand crops that were picked by visual review.
Candidates were drawn from random crop sheets of the pre-annotations; the CLIP defect hint was NOT useful for this."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT = Path(__file__).resolve().parent.parent; SRC = Path("/workspace/rag_civitai")
rows = {r["id"]: r for r in map(json.loads, open(ROOT / "data/preannotations.jsonl"))}
EX = json.load(open(ROOT / "docs/guide_examples.json"))
font = ImageFont.load_default(size=17); big = ImageFont.load_default(size=24)

def crop(iid, j, size=230):
    r = rows[iid]; im = Image.open(SRC / r["path"]).convert("RGB"); x0, y0, x1, y1 = r["boxes"][j]["xyxy"]
    p = 0.12 * max(x1 - x0, y1 - y0)
    c = im.crop((max(0, x0 - p), max(0, y0 - p), min(im.width, x1 + p), min(im.height, y1 + p)))
    s = min(size / c.width, size / c.height); return c.resize((int(c.width * s), int(c.height * s)), Image.LANCZOS)

def sheet(groups, out, cols=8, tw=236, th=270):
    H = sum(40 + ((len(g["items"]) + cols - 1) // cols) * th for g in groups)
    C = Image.new("RGB", (cols * tw, H), (24, 24, 24)); d = ImageDraw.Draw(C); y = 0
    for g in groups:
        d.text((8, y + 8), g["title"], fill=tuple(g["color"]), font=big); y += 40
        for k, it in enumerate(g["items"]):
            c = crop(it["id"], it["box"]); x = (k % cols) * tw; yy = y + (k // cols) * th
            C.paste(c, (x + 3 + (230 - c.width) // 2, yy + 3 + (230 - c.height) // 2))
            d.rectangle([x + 1, yy + 1, x + tw - 3, yy + 234], outline=tuple(g["color"]), width=2)
            d.text((x + 6, yy + 240), it["note"], fill=(230, 230, 230), font=font)
        y += ((len(g["items"]) + cols - 1) // cols) * th
    C.save(out, quality=88)

sheet(EX["ok_bad"], ROOT / "docs/examples_ok_bad.jpg")
sheet(EX["edge"], ROOT / "docs/examples_edge_cases.jpg", cols=6)
