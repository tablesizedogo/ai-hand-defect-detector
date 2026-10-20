"""Step 1: crop every pre-annotated hand box (square, 30% padding) -> autolabel_data/crops/<key>.jpg + manifest.jsonl."""
import json
from pathlib import Path
from PIL import Image
R = Path(__file__).resolve().parents[1]; IMG = Path('/workspace/rag_civitai')
OUT = R / 'autolabel_data'; (OUT / 'crops').mkdir(parents=True, exist_ok=True)
PAD = 1.3; MAXS = 448
with open(OUT / 'manifest.jsonl', 'w') as mf:
    n = 0
    for line in open(R / 'data/preannotations.jsonl'):
        r = json.loads(line); im = None
        for j, b in enumerate(r['boxes']):
            if im is None: im = Image.open(IMG / r['path']).convert('RGB')
            x0, y0, x1, y1 = b['xyxy']; w, h = x1 - x0, y1 - y0
            s = max(w, h) * PAD / 2; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            box = [int(max(0, cx - s)), int(max(0, cy - s)), int(min(im.width, cx + s)), int(min(im.height, cy + s))]
            c = im.crop(box); c.thumbnail((MAXS, MAXS))
            key = f"{r['id']}_{j}"; c.save(OUT / f'crops/{key}.jpg', quality=92)
            mf.write(json.dumps(dict(key=key, id=r['id'], box_idx=j, family=r['family'], baseModel=r['baseModel'], style=r['style'],
                user=r['user'], xyxy=b['xyxy'], crop_xyxy=box, det_conf=b['score'], det_src=b['src'], short_side=round(min(w, h), 1),
                crop_wh=list(c.size))) + '\n'); n += 1
print('crops', n)
