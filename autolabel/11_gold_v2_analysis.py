"""Step 11: extra analysis of gold v2 (complements 07): enrichment hit rate per stratum, weighted vs unweighted AUC,
the v2 teacher on the crops it covers (4,325-crop snapshot), gate check on the gate_not_hand stratum and the nohand answers,
the BAD crops and the top false alarms of the v3 teacher. Writes autolabel_data/gold_check_v2/analysis.{md,json}
and a contact sheet of the BAD crops + top false alarms (false_alarms_sheet.jpg)."""
import json, re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from common import D, read_jsonl
G = D / 'gold_check_v2'; man = json.load(open(G / 'manifest.json')); items = {it['num']: it for it in man['items']}
lab = {}
for line in open(G / 'answers.txt'):
    m = re.match(r'\s*(bad|nohand|skip)\s*:(.*)', line.split('#')[0], re.I)
    if m:
        for n in map(int, re.findall(r'\d+', m.group(2))): lab[n] = m.group(1).lower()
for n in items: lab.setdefault(n, 'ok')
fu = read_jsonl(D / 'fused.jsonl'); fu2 = read_jsonl(D / 'v2_gate095_snap4325/fused.jsonl')
st3 = read_jsonl(D / 'student_clip_l14_logreg.jsonl'); st2 = read_jsonl(D / 'v2_gate095_snap4325/student_clip_l14_logreg.jsonl')
L = ['# Gold v2: extra analysis', '']
# 1. enrichment hit rate
L += ['## Labels per stratum (raw counts)', '', '| stratum | n | weight per item | OK | BAD | nohand | skip | raw BAD rate among judged hands |', '|---|---|---|---|---|---|---|---|']
strata = {}
for n, it in items.items(): strata.setdefault(it['stratum'], []).append(n)
res = dict(strata={})
for s in ['low_teacher', 'rest_q1', 'rest_q2', 'rest_q3', 'rest_q4', 'gate_not_hand']:
    ns = strata[s]; c = {k: sum(lab[n] == k for n in ns) for k in ('ok', 'bad', 'nohand', 'skip')}
    rate = c['bad'] / max(1, c['ok'] + c['bad']); res['strata'][s] = dict(n=len(ns), weight=items[ns[0]]['weight'], **c, bad_rate=rate)
    L.append(f"| {s} | {len(ns)} | {items[ns[0]]['weight']:.1f} | {c['ok']} | {c['bad']} | {c['nohand']} | {c['skip']} | {rate:.1%} |")
H = [n for n in items if lab[n] in ('ok', 'bad')]; y = np.array([lab[n] == 'ok' for n in H], int); w = np.array([items[n]['weight'] for n in H])
pw = 1 - np.average(y, weights=w); L += ['', f"Weighted BAD prevalence among gate-passed hands (population estimate): {pw:.1%}; raw share in the sample: {(y == 0).mean():.1%}.", '']
res['bad_prevalence_weighted'] = float(pw)
# 2. weighted vs unweighted AUC, and v2 teacher on its covered subset
K = [items[n]['key'] for n in H]
S = {'v3 teacher': np.array([fu[k]['teacher_ok'] for k in K]), 'HADM alone (−score)': -np.array([fu[k]['hadm_hand'] for k in K]),
     'Gemma p_ok': np.array([fu[k]['gemma_p_ok'] for k in K]), 'det_conf': np.array([fu[k]['det_conf'] for k in K]),
     'CLIP-L student v3': np.array([st3[k]['student_p_ok'] for k in K]), 'CLIP-L student v2': np.array([st2[k]['student_p_ok'] for k in K]),
     'v2 teacher (Gemma+det+CLIP)': np.array([fu2[k]['teacher_ok'] if fu2.get(k) and fu2[k]['teacher_ok'] is not None else np.nan for k in K])}
cov = ~np.isnan(S['v2 teacher (Gemma+det+CLIP)'])
rng = np.random.default_rng(0)
def boot(x, m):
    io, ib = np.where(m & (y == 1))[0], np.where(m & (y == 0))[0]; out = []
    for _ in range(2000):
        ix = np.concatenate([rng.choice(io, len(io)), rng.choice(ib, len(ib))]); b = 1 - y[ix]
        fpr, tpr, _ = roc_curve(b, -x[ix], sample_weight=w[ix])
        out.append((roc_auc_score(b, -x[ix], sample_weight=w[ix]), average_precision_score(b, -x[ix], sample_weight=w[ix]),
                    tpr[fpr <= .05 + 1e-12].max(), tpr[fpr <= .10 + 1e-12].max()))
    b = 1 - y[m]; fpr, tpr, _ = roc_curve(b, -x[m], sample_weight=w[m])
    pt = (roc_auc_score(b, -x[m], sample_weight=w[m]), average_precision_score(b, -x[m], sample_weight=w[m]), tpr[fpr <= .05 + 1e-12].max(), tpr[fpr <= .10 + 1e-12].max())
    lo, hi = np.percentile(out, 2.5, 0), np.percentile(out, 97.5, 0); return [(float(p), float(l), float(h)) for p, l, h in zip(pt, lo, hi)]
L += [f'## Same scores on the {int(cov.sum())} of {len(H)} gold-v2 hands covered by the v2 teacher (4,325-crop VLM snapshot; {int((cov & (y == 0)).sum())} BAD)', '',
      'Weighted, label-stratified bootstrap (2000). The v2 teacher has no score for the other crops, so it is missing from eval.md.', '',
      '| score | AUC | AP (BAD) | recall @5% FPR | recall @10% FPR | unweighted AUC (all v2 hands) |', '|---|---|---|---|---|---|']
res['covered_subset'] = {}
for s, x in S.items():
    r = boot(x, cov); res['covered_subset'][s] = r
    uw = roc_auc_score(1 - y, -x) if not np.isnan(x).any() else float('nan')
    L.append(f"| {s} | " + ' | '.join(f'{p:.3f} [{l:.2f}–{h:.2f}]' for p, l, h in r) + f" | {uw:.3f} |")
# within-stratum AUC (teacher's ranking inside the enriched stratum, no weighting needed)
L += ['', '## AUC inside the low_teacher stratum only (60 crops from the bottom 10% of the v3 teacher; unweighted)', '', '| score | AUC |', '|---|---|']
ms = np.array([items[n]['stratum'] == 'low_teacher' for n in H])
for s, x in S.items():
    if not np.isnan(x[ms]).any() and len(set(y[ms])) == 2: L.append(f"| {s} | {roc_auc_score(1 - y[ms], -x[ms]):.3f} |")
# 3. gate
L += ['', '## Hand gate on gold v2', '', '| # | stratum | user label | Gemma p_hand | det_conf | gate says |', '|---|---|---|---|---|---|']
for n in sorted(items):
    it = items[n]; f = fu[it['key']]
    if it['stratum'] == 'gate_not_hand' or lab[n] == 'nohand':
        L.append(f"| {n} | {it['stratum']} | {lab[n]} | {f['gemma_p_hand']:.3f} | {f['det_conf']:.2f} | {'hand' if f['is_hand'] else 'not a hand'} |")
# 4. BAD crops and false alarms
pct = lambda x, i: float((x < x[i]).mean())
L += ['', '## BAD crops: percentile rank among gold-v2 hands (0 = flagged first)', '', '| # | stratum | style | ' + ' | '.join(S) + ' |', '|---' * (3 + len(S)) + '|']
for n in [n for n in H if lab[n] == 'bad']:
    i = H.index(n); L.append(f"| {n} | {items[n]['stratum']} | {items[n]['style']} | " + ' | '.join('–' if np.isnan(x[i]) else f'{pct(x, i):.2f}' for x in S.values()) + ' |')
t = S['v3 teacher']; order = np.argsort(t); fa = [H[i] for i in order if y[i] == 1][:10]
L += ['', '## Top 10 false alarms of the v3 teacher (OK hands it ranks most suspicious)', '', '| # | stratum | style | teacher_ok | HADM | Gemma p_ok | det_conf |', '|---|---|---|---|---|---|---|']
for n in fa:
    f = fu[items[n]['key']]; L.append(f"| {n} | {items[n]['stratum']} | {items[n]['style']} | {f['teacher_ok']:.3f} | {f['hadm_hand']:.2f} | {f['gemma_p_ok']:.2f} | {f['det_conf']:.2f} |")
nb = sum(1 for i in order[:len(order) // 4] if y[i] == 0)
L += ['', f"Of the 25% most suspicious gold-v2 hands by the v3 teacher ({len(order) // 4}), {nb} are BAD."]
res['false_alarms'] = fa
# contact sheet: BAD then false alarms
sel = [(n, 'BAD') for n in H if lab[n] == 'bad'] + [(n, 'FA') for n in fa]
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30); T = 260; C = 6; rows = (len(sel) + C - 1) // C
sh = Image.new('RGB', (C * T, rows * T), (24, 24, 24)); d = ImageDraw.Draw(sh)
for i, (n, tg) in enumerate(sel):
    c = Image.open(D / f"crops/{items[n]['key']}.jpg").convert('RGB'); c.thumbnail((T - 8, T - 8)); x, yy = (i % C) * T, (i // C) * T
    sh.paste(c, (x + 4, yy + 4)); d.rectangle([x + 4, yy + 4, x + 120, yy + 44], fill=(0, 0, 0)); d.text((x + 8, yy + 8), f'{n} {tg}', fill=(255, 80, 80) if tg == 'BAD' else (255, 230, 0), font=font)
sh.save(G / 'bad_and_false_alarms.jpg', quality=88)
json.dump(res, open(G / 'analysis.json', 'w'), indent=1); open(G / 'analysis.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
