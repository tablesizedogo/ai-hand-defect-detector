"""Step 9.1: build the HADM input set for the 150 gold crops.

For every source image behind a gold crop, fetch the full-resolution original from the Civitai CDN
(`original=true` variant of the `download_url` stored in data/preannotations.jsonl), convert to RGB JPEG (q=95; HADM expects
JPEG input) with the long side capped at 1024 px (HADM-L resizes the long side to 1024 anyway). If the download fails or the
aspect ratio does not match the local copy, fall back to the local 512-px image (logged per image).
Crop boxes (`xyxy` = detector hand box, `crop_xyxy` = padded crop) are rescaled from 512-px coordinates to the saved image.
Only the already SFW-filtered corpus images behind the gold crops are used.
Output: autolabel_data/hadm/input/{images/<id>.jpg, boxes.json, dataset-metadata.json}
`--all`: every source image behind the 5,604 crops of autolabel_data/manifest.jsonl -> autolabel_data/hadm/input_all/
(threaded download; images already fetched for the gold run are reused).
"""
import argparse, json, io, shutil, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import R, D

ap = argparse.ArgumentParser(); ap.add_argument('--all', action='store_true'); ap.add_argument('--workers', type=int, default=8); A = ap.parse_args()
OUT = D / ('hadm/input_all' if A.all else 'hadm/input'); (OUT / 'images').mkdir(parents=True, exist_ok=True)
meta = {json.loads(l)['key']: json.loads(l) for l in open(D / 'manifest.jsonl')}
if A.all: gold = [dict(key=k, num=None, style=m['style'], family=m['family']) for k, m in meta.items()]
else: gold = json.load(open(D / 'gold_check/manifest.json'))['items']
GOLD_DIR = D / 'hadm/input/images'
prev = json.load(open(D / 'hadm/input/boxes.json'))['images'] if A.all and (D / 'hadm/input/boxes.json').exists() else {}
pre = {json.loads(l)['id']: json.loads(l) for l in open(R / 'data/preannotations.jsonl')}
MAXS = 1024

def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r: return r.read()

def one(iid):
    if f'{iid}.jpg' in prev and (GOLD_DIR / f'{iid}.jpg').exists():
        shutil.copy(GOLD_DIR / f'{iid}.jpg', OUT / f'images/{iid}.jpg'); return iid, prev[f'{iid}.jpg']
    p = pre[iid]; local = Image.open(Path('/workspace/rag_civitai') / p['path']).convert('RGB'); W5, H5 = local.size
    src, im = 'local512', local
    url = p['download_url'].replace('/width=512,optimized=true/', '/original=true/').replace('/width=512/', '/original=true/')
    if 'original=true' in url:
        for attempt in range(3):
            try:
                o = Image.open(io.BytesIO(fetch(url))).convert('RGB')
                if abs(o.width / o.height - W5 / H5) < 0.01 and o.width > W5: src, im = 'civitai_original', o
                else: print('aspect/size mismatch', iid, o.size, (W5, H5))
                break
            except Exception as e:
                print('retry', iid, e); time.sleep(2)
    orig_size = im.size
    if max(im.size) > MAXS:
        s = MAXS / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    im.save(OUT / f'images/{iid}.jpg', quality=95)
    return iid, dict(src=src, orig_size=orig_size, size=im.size, size512=(W5, H5), sx=im.width / W5, sy=im.height / H5)

ids = list(dict.fromkeys(int(it['key'].split('_')[0]) for it in gold)); imgs = {}
with ThreadPoolExecutor(A.workers) as ex:
    for i, (iid, g) in enumerate(ex.map(one, ids)):
        imgs[iid] = g
        if i % 100 == 0 or not A.all: print(i + 1, len(ids), iid, g['src'], g['orig_size'], '->', g['size'], flush=True)

crops = {}
for it in gold:
    k = it['key']; iid = int(k.split('_')[0]); g = imgs[iid]; m = meta[k]
    sc = lambda b: [b[0] * g['sx'], b[1] * g['sy'], b[2] * g['sx'], b[3] * g['sy']]
    crops[k] = dict(num=it.get('num'), image=f'{iid}.jpg', style=it['style'], family=it['family'],
                    hand_xyxy=sc(m['xyxy']), crop_xyxy=sc(m['crop_xyxy']))
json.dump(dict(images={f'{k}.jpg': v for k, v in imgs.items()}, crops=crops), open(OUT / 'boxes.json', 'w'), indent=1)
n = sum(v['src'] == 'civitai_original' for v in imgs.values())
print(f'{len(imgs)} images ({n} Civitai originals, {len(imgs) - n} local 512-px fallbacks), {len(crops)} crops')
