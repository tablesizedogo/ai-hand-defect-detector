"""Step 6: build the ~150-crop gold-check set for the user and render compact numbered contact sheets.

Strata: 10 deciles of the score (student if --score points to a student jsonl, else teacher_ok) among is_hand crops,
balanced over styles (photo/anime/art) inside each decile, plus N_GATE crops the VLM gate called "not a hand".
Every item keeps a sampling weight = stratum size / items sampled from it, so metrics can be re-weighted to the
real population (the sample is deliberately enriched with low-score crops).
Output: autolabel_data/gold_check/{manifest.json, sheet_01.jpg ...}. Numbers 1..N are global across sheets, order is shuffled,
no scores are shown. The 40 study crops are excluded.
Usage: python autolabel/06_goldset.py [--n-hand 125] [--n-gate 25] [--allow-partial] [--score autolabel_data/student_clip_l14_logreg.jsonl]
"""
import argparse, json, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import R, D, read_jsonl
ap = argparse.ArgumentParser(); ap.add_argument('--n-hand', type=int, default=125); ap.add_argument('--n-gate', type=int, default=25)
ap.add_argument('--per-sheet', type=int, default=25); ap.add_argument('--tile', type=int, default=230)
ap.add_argument('--score', default=None); ap.add_argument('--allow-partial', action='store_true'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--out', default='gold_check')
a = ap.parse_args(); rng = random.Random(a.seed)
fu = read_jsonl(D / 'fused.jsonl'); keys = list(fu)
if not all(fu[k]['vlm_done'] for k in keys) and not a.allow_partial:
    raise SystemExit('VLM job not finished: rerun 04_fuse.py when done, or pass --allow-partial (gate unknown crops count as hands)')
excl = set(json.load(open(R / 'auto/gold40.json'))) if (R / 'auto/gold40.json').exists() else set()
sc = {k: fu[k]['teacher_ok'] for k in keys}; sname = 'teacher_ok'
if a.score:
    st = read_jsonl(a.score); sc = {k: st[k]['student_p_ok'] for k in keys}; sname = a.score
hands = [k for k in keys if fu[k]['is_hand'] is not False and k not in excl]
gate = [k for k in keys if fu[k]['is_hand'] is False and k not in excl]
items = []
qs = np.quantile([sc[k] for k in hands], np.linspace(0, 1, 11))
per = [a.n_hand // 10 + (1 if i < a.n_hand % 10 else 0) for i in range(10)]
for i in range(10):
    s = [k for k in hands if (sc[k] >= qs[i]) and (sc[k] < qs[i + 1] or i == 9)]
    by = {}
    for k in s: by.setdefault(fu[k]['style'], []).append(k)
    for v in by.values(): rng.shuffle(v)
    pick = []; styles = sorted(by)
    while len(pick) < min(per[i], len(s)):
        for st_ in styles:
            if by[st_] and len(pick) < per[i]: pick.append(by[st_].pop())
    items += [dict(key=k, stratum=f'decile{i + 1}', weight=len(s) / len(pick)) for k in pick]
if gate and a.n_gate:
    pick = rng.sample(gate, min(a.n_gate, len(gate))); items += [dict(key=k, stratum='gate_not_hand', weight=len(gate) / len(pick)) for k in pick]
rng.shuffle(items)
for n, it in enumerate(items, 1): it.update(num=n, style=fu[it['key']]['style'], family=fu[it['key']]['family'], sheet=(n - 1) // a.per_sheet + 1)
out = D / a.out; out.mkdir(exist_ok=True)
for f in out.glob('sheet_*.jpg'): f.unlink()
try: font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
except Exception: font = ImageFont.load_default()
T = a.tile; C = 5
for sidx in range(0, len(items), a.per_sheet):
    chunk = items[sidx:sidx + a.per_sheet]; rows = (len(chunk) + C - 1) // C
    sh = Image.new('RGB', (C * T, rows * T), (24, 24, 24)); d = ImageDraw.Draw(sh)
    for i, it in enumerate(chunk):
        c = Image.open(D / f"crops/{it['key']}.jpg").convert('RGB'); f = (T - 8) / max(c.size)
        c = c.resize((max(1, int(c.width * f)), max(1, int(c.height * f))), Image.LANCZOS)
        x, y = (i % C) * T, (i // C) * T; sh.paste(c, (x + 4 + (T - 8 - c.width) // 2, y + 4 + (T - 8 - c.height) // 2))
        d.rectangle([x + 4, y + 4, x + 52, y + 32], fill=(0, 0, 0)); d.text((x + 8, y + 6), str(it['num']), fill=(255, 230, 0), font=font)
    sh.save(out / f'sheet_{sidx // a.per_sheet + 1:02d}.jpg', quality=88)
json.dump(dict(score=sname, n=len(items), allow_partial=a.allow_partial, items=items), open(out / 'manifest.json', 'w'), indent=1)
open(out / 'answers_TEMPLATE.txt', 'w').write(
    '# Numbers are global (1..%d). Everything not listed = OK hand.\n# bad   = anatomically broken hand (extra/missing/fused fingers, melted, blob)\n'
    '# nohand = not a human hand (paw, object, face, nothing)\n# skip  = cannot judge (too small/blurry/occluded)\nbad:\nnohand:\nskip:\n' % len(items))
print(f'{len(items)} items ({sum(it["stratum"] != "gate_not_hand" for it in items)} hand-strata, '
      f'{sum(it["stratum"] == "gate_not_hand" for it in items)} gate) -> {out}, sheets: {(len(items) + a.per_sheet - 1) // a.per_sheet}')
