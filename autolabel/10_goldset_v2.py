"""Step 10: second gold-check round (gold v2): ~100 NEW crops (not in gold v1, not study-40), enriched for likely-BAD.

Selected with the v3 teacher (HADM + Gemma + detector conf, fused.jsonl) BEFORE the v3 students are trained, and 05_silver_train.py
excludes gold v2 from training, so gold v2 stays an honest test for the students. (The v3 teacher itself was chosen on gold v1,
not on v2, so v2 is honest for the teacher too.)
Strata (is_hand crops, gate p_hand >= 0.95; style-balanced picks inside each stratum):
  low_teacher   N_LOW (60) from the lowest 10% of teacher_ok          -> enriched for BAD
  rest_q1..q4   N_REST (32) from the other 90%, in 4 equal-count bins of teacher_ok
  gate_not_hand N_GATE (8) from crops the gate calls "not a hand"     -> checks the gate
Each item has weight = stratum size / picks, so 07 can re-weight to the population (all crops outside gold v1 / study-40).
Output: autolabel_data/gold_check_v2/{manifest.json, sheet_01..04.jpg, answers_TEMPLATE.txt}; numbers 1..100, shuffled, no scores shown.
--replace k1,k2 swaps SFW-rejected crops in place for another crop of the same stratum and style (numbering unchanged).
"""
import argparse, json, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import R, D, read_jsonl
ap = argparse.ArgumentParser(); ap.add_argument('--n-low', type=int, default=60); ap.add_argument('--n-rest', type=int, default=32)
ap.add_argument('--n-gate', type=int, default=8); ap.add_argument('--low-q', type=float, default=0.10)
ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='gold_check_v2'); ap.add_argument('--tile', type=int, default=300)
ap.add_argument('--replace', default=''); a = ap.parse_args()
rng = random.Random(a.seed); swap_rng = random.Random(1000 + a.seed)
fu = read_jsonl(D / 'fused.jsonl'); meta = json.load(open(D / 'fused_meta.json'))
assert meta.get('teacher') == 'hadm_gemma_det', 'run 04_fuse.py --teacher hadm_gemma_det first'
excl = {it['key'] for it in json.load(open(D / 'gold_check/manifest.json'))['items']}
if (R / 'auto/gold40.json').exists(): excl |= set(json.load(open(R / 'auto/gold40.json')))
keys = [k for k in fu if k not in excl]
hands = sorted((k for k in keys if fu[k]['is_hand']), key=lambda k: fu[k]['teacher_ok'])
gate = [k for k in keys if fu[k]['is_hand'] is False]
nl = int(round(a.low_q * len(hands))); low, rest = hands[:nl], hands[nl:]
strata = [('low_teacher', low, a.n_low)] + [(f'rest_q{i + 1}', list(b), a.n_rest // 4) for i, b in enumerate(np.array_split(rest, 4))] + [('gate_not_hand', gate, a.n_gate)]
def pick_balanced(pool, n):
    by = {}
    for k in pool: by.setdefault(fu[k]['style'], []).append(k)
    for v in by.values(): rng.shuffle(v)
    out = []
    while len(out) < min(n, len(pool)):
        for st in sorted(by):
            if by[st] and len(out) < n: out.append(by[st].pop())
    return out
items = []
for nm, pool, n in strata:
    pk = pick_balanced(pool, n); items += [dict(key=k, stratum=nm, weight=len(pool) / len(pk), _pool=pool) for k in pk]
rej = {k for k in a.replace.split(',') if k}; used = {it['key'] for it in items}; swapped = []
for it in items:
    if it['key'] in rej:
        cand = sorted(k for k in it['_pool'] if k not in used and k not in rej and fu[k]['style'] == fu[it['key']]['style'])
        new = swap_rng.choice(cand); swapped.append((it['key'], new)); used.add(new); it['key'] = new
for it in items: it.pop('_pool')
rng.shuffle(items)
for n, it in enumerate(items, 1): it.update(num=n, style=fu[it['key']]['style'], family=fu[it['key']]['family'], sheet=(n - 1) // 25 + 1)
out = D / a.out; out.mkdir(exist_ok=True)
for f in out.glob('sheet_*.jpg'): f.unlink()
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 40)
T, C = a.tile, 5
for si in range(0, len(items), 25):
    chunk = items[si:si + 25]; rows = (len(chunk) + C - 1) // C
    sh = Image.new('RGB', (C * T, rows * T), (24, 24, 24)); d = ImageDraw.Draw(sh)
    for i, it in enumerate(chunk):
        c = Image.open(D / f"crops/{it['key']}.jpg").convert('RGB'); f = (T - 8) / max(c.size)
        c = c.resize((max(1, int(c.width * f)), max(1, int(c.height * f))), Image.LANCZOS)
        x, y = (i % C) * T, (i // C) * T; sh.paste(c, (x + 4 + (T - 8 - c.width) // 2, y + 4 + (T - 8 - c.height) // 2))
        bb = d.textbbox((0, 0), str(it['num']), font=font); bw, bh = bb[2] - bb[0], bb[3] - bb[1]
        d.rectangle([x + 4, y + 4, x + 4 + bw + 16, y + 4 + bh + 16], fill=(0, 0, 0), outline=(255, 230, 0), width=2)
        d.text((x + 12 - bb[0], y + 12 - bb[1]), str(it['num']), fill=(255, 230, 0), font=font)
    sh.save(out / f'sheet_{si // 25 + 1:02d}.jpg', quality=88)
json.dump(dict(round='v2', score='teacher_ok (v3: hadm+gemma+det)', n=len(items), seed=a.seed, low_q=a.low_q, population=len(keys),
               excluded='gold v1 + study-40', sfw_rejected=sorted(rej), swaps=swapped, items=items), open(out / 'manifest.json', 'w'), indent=1)
open(out / 'answers_TEMPLATE.txt', 'w').write(
    '# Gold check v2. Numbers are 1..%d (sheets 1-4). Everything not listed = OK hand.\n# bad   = anatomically broken hand (extra/missing/fused fingers, melted, blob)\n'
    '# nohand = not a human hand (paw, object, face, nothing)\n# skip  = cannot judge (too small/blurry/occluded)\nbad:\nnohand:\nskip:\n' % len(items))
from collections import Counter
print(len(items), Counter(it['stratum'] for it in items), Counter(it['style'] for it in items), 'swaps', swapped)
