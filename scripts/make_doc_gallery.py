"""Example gallery for the docs: hand crops scored by the CLIP fine-tune (ft3), with the true label and the verdict.

Sources: Civitai gold crops (local, unpublished inputs), AbHuman val and Distorted-5K (public, fetched with HTTP range
requests and cut exactly as in the Kaggle prep: 1.5x square crop of the hand box, 256 px). No HAD images.
Every crop and its source image passed the SFW rule used for the training data (scripts/check_gallery_sfw.py:
Falconsai P(nsfw) < 0.2 and CLIP-B/32 'suggestive' < 0.5) and a visual check.
Scores: external_experiments/results/ft3/ft_scores_gold_cpu.json and results/ft3_score/ft_scores_{ab,d5k}.json.
Labels: results/fa/gold_table.json (gold) and results/prep/hands.jsonl (AbHuman, Distorted-5K).

Run: python scripts/make_doc_gallery.py   (needs Pillow and remotezip)
"""
import io, json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EX = os.path.join(ROOT, "external_experiments")
CACHE = os.path.join(EX, "results", "gallery_crops")
THR = 0.83  # flag threshold: 10% false-alarm rate on gold (see the long doc)

# (key, source, short description of what the crop shows[, verdict override after manual review])
ITEMS = [
    ("129846139_2", "gold", "blurred, shapeless fist"),
    ("ab_val_8363_0", "ab", "tangled fingers, comic style"),
    ("d5k_eval_708837483_0", "d5k", "hand writing with a pen"),
    ("142337418_1", "gold", "open palm behind glass"),
    ("d5k_eval_708837421_0", "d5k", "hand cutting with a knife"),
    ("142436825_3", "gold", "claws, too many long fingers"),
    ("142954633_1", "gold", "plate gauntlet on a sword"),
    ("ab_val_humangeneral9439_0", "ab", "blurred hand on a dog"),
    ("142652781_1", "gold", "fingers on a guitar neck"),
    ("ab_val_7355_0", "ab", "has six fingers", "label error"),  # AbHuman says normal; the hand has six fingers
]
SMALL = ["129846139_2", "d5k_eval_708837483_0", "142337418_1", "142436825_3", "142652781_1", "ab_val_7355_0"]
SRC_NAME = {"gold": "Civitai gold", "ab": "AbHuman", "d5k": "Distorted-5K"}
ZIPS = {"ab": "https://huggingface.co/datasets/Enderfga/HumanRefiner/resolve/main/val.zip",
        "d5k": "https://huggingface.co/datasets/xgklndsgkl/Distorted-5K/resolve/main/distorted-5k.zip"}


def sqcrop(im, b, pad, size):  # same as kaggle/k1_prep/hdx_prep.py
    x0, y0, x1, y1 = b[:4]; s = max(x1 - x0, y1 - y0) * pad / 2; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return im.crop((int(max(0, cx - s)), int(max(0, cy - s)), int(min(im.width, cx + s)), int(min(im.height, cy + s)))).resize((size, size), Image.BICUBIC)


def load():
    gold = {r["key"]: r for r in json.load(open(f"{EX}/results/fa/gold_table.json"))}
    gs = json.load(open(f"{EX}/results/ft3/ft_scores_gold_cpu.json"))
    es = {}
    for s in ("ab", "d5k"):
        es.update(json.load(open(f"{EX}/results/ft3_score/ft_scores_{s}.json")))
    want = {k for k, s, *_ in ITEMS if s != "gold"}
    hands = {}
    for line in open(f"{EX}/results/prep/hands.jsonl"):
        r = json.loads(line)
        if r["key"] in want: hands[r["key"]] = r
    os.makedirs(CACHE, exist_ok=True); zips = {}
    out = []
    for k, s, note, *override in ITEMS:
        if s == "gold":
            im = Image.open(f"{EX}/kaggle/inputs/gold_crops/{k}_p15.jpg").convert("RGB")
            y, p = gold[k]["y"], gs["gold_" + k]
        else:
            h = hands[k]; y, p = int(h["label"] == "bad"), es[k]; path = f"{CACHE}/{k}.jpg"
            if not os.path.exists(path):
                from remotezip import RemoteZip
                z = zips.setdefault(s, RemoteZip(ZIPS[s]))
                member = f"val/images/{h['image']}.jpg" if s == "ab" else f"distorted-5k/{h['image']}"
                full = Image.open(io.BytesIO(z.read(member))).convert("RGB")
                if max(full.size) > 1024: full.thumbnail((1024, 1024), Image.BICUBIC)
                full.save(path[:-4] + "__full.jpg", quality=92); sqcrop(full, h["xyxy"], 1.5, 256).save(path, quality=92)
            im = Image.open(path).convert("RGB")
        flag = p >= THR
        verdict = override[0] if override else ("correct" if flag == bool(y) else ("false alarm" if flag else "missed"))
        out.append(dict(key=k, src=s, note=note, y=y, p=p, verdict=verdict, im=im))
    return out


COL = {"correct": "#2e7d32", "false alarm": "#c62828", "missed": "#ef6c00", "label error": "#ffc107"}
TXT = {"label error": "#111"}


def sheet(items, ncol, path, tile=230):
    f = lambda sz, b=False: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), sz)
    cap = 78; pad = 10; nrow = (len(items) + ncol - 1) // ncol
    W, H = ncol * (tile + pad) + pad, nrow * (tile + cap + pad) + pad
    img = Image.new("RGB", (W, H), "white"); d = ImageDraw.Draw(img)
    for i, it in enumerate(items):
        x = pad + (i % ncol) * (tile + pad); y = pad + (i // ncol) * (tile + cap + pad)
        img.paste(it["im"].resize((tile, tile), Image.BICUBIC), (x, y))
        c = COL[it["verdict"]]
        d.rectangle([x, y + tile, x + tile - 1, y + tile + 22], fill=c)
        d.text((x + 6, y + tile + 3), it["verdict"].upper(), font=f(13, True), fill=TXT.get(it["verdict"], "white"))
        d.text((x + 4, y + tile + 27), (f"labelled normal, P(bad) {it['p']:.2f}" if it["verdict"] == "label error" else f"{'bad' if it['y'] else 'normal'} hand, P(bad) {it['p']:.2f}"), font=f(13), fill="#111")
        d.text((x + 4, y + tile + 44), it["note"], font=f(12), fill="#444")
        d.text((x + 4, y + tile + 60), SRC_NAME[it["src"]], font=f(11), fill="#777")
    img.save(path, quality=90)
    print(path, img.size)


if __name__ == "__main__":
    items = load()
    for it in items: print(it["key"], it["src"], it["y"], round(it["p"], 3), it["verdict"])
    sheet(items, 5, os.path.join(ROOT, "docs/experiments/img/ft3_gallery.jpg"))
    by = {it["key"]: it for it in items}
    sheet([by[k] for k in SMALL], 3, os.path.join(ROOT, "docs/img/ft3_gallery_small.jpg"))
