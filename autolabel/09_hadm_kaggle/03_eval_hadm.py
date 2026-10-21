"""Step 9.3: evaluate HADM-L 'hand' artifact detections as a hand-quality score on the user's gold check.

Input: autolabel_data/hadm/out_gold/hadm_detections.json (from the Kaggle kernel) + autolabel_data/hadm/input/boxes.json.
For each gold crop, score_bad = max score of HADM detections of class `hand` (or all classes) that match the crop under a
matching rule; 0 if none matches. Rules (boxes in the coordinates of the image HADM saw):
  iou_hand>=t      IoU(detection, detector hand box) >= t
  iod_crop>=t      area(det ∩ padded crop box) / area(det) >= t   (artifact box mostly inside the crop)
  soft_crop        max over dets of score * area(det ∩ crop)/area(det)
  whole_image      max over all dets in the image (no localisation)
Score used for metrics = -score_bad (higher = more OK), so the tables compare 1:1 with eval.md.
Protocol (eval subset = gold OK + BAD hands; stratified-sampling weights; label-stratified bootstrap with the same seed and
resample construction; AUC, AP(BAD), recall of BAD at 5%/10% FPR) is identical to 07_ingest_eval.py.
Also: per-style split (anime / art / photo) with its own label-stratified bootstrap, exploratory rank fusions with
Gemma / detector confidence (selected on gold -> optimistic), per-BAD ranks.
Writes autolabel_data/gold_check/eval_hadm.{json,md} and autolabel_data/hadm/hadm_gold_scores.jsonl.
"""
import argparse, json, re, sys
import numpy as np
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import D, read_jsonl
ap = argparse.ArgumentParser(); ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--det', default=str(D / 'hadm/out_gold/hadm_detections.json')); a = ap.parse_args()
G = D / 'gold_check'
man = json.load(open(G / 'manifest.json')); items = {it['num']: it for it in man['items']}
lab = {}
for line in open(G / 'answers.txt'):
    line = line.split('#')[0]; m = re.match(r'\s*(bad|nohand|skip)\s*:(.*)', line, re.I)
    if m:
        for n in map(int, re.findall(r'\d+', m.group(2))): lab[n] = m.group(1).lower()
for n in items: lab.setdefault(n, 'ok')
boxes = json.load(open(D / 'hadm/input/boxes.json')); det = json.load(open(a.det)); dmeta, det = det['meta'], det['detections']
fu = read_jsonl(D / 'fused.jsonl'); stu = read_jsonl(D / 'student_clip_l14_logreg.jsonl')

def inter(b, c): return max(0, min(b[2], c[2]) - max(b[0], c[0])) * max(0, min(b[3], c[3]) - max(b[1], c[1]))
def area(b): return max(0, b[2] - b[0]) * max(0, b[3] - b[1])
def iou(b, c): i = inter(b, c); return i / max(1e-9, area(b) + area(c) - i)
def hadm_scores(key):
    c = boxes['crops'][key]; d = det[c['image']]; out = {}
    for cls_nm, keep in (('hand', lambda k: k == 'hand'), ('any', lambda k: True)):
        D_ = [(b, s) for b, s, k in zip(d['bbox'], d['score'], d['class']) if keep(k)]
        mx = lambda vals: max(vals, default=0.0)
        for t in (0.1, 0.3, 0.5): out[f'{cls_nm} iou_hand>={t}'] = mx([s for b, s in D_ if iou(b, c['hand_xyxy']) >= t])
        for t in (0.3, 0.5): out[f'{cls_nm} iod_crop>={t}'] = mx([s for b, s in D_ if inter(b, c['crop_xyxy']) / max(1e-9, area(b)) >= t])
        out[f'{cls_nm} soft_crop'] = mx([s * inter(b, c['crop_xyxy']) / max(1e-9, area(b)) for b, s in D_])
        out[f'{cls_nm} whole_image'] = mx([s for b, s in D_])
    return out

H = [n for n in items if lab[n] in ('ok', 'bad')]
y = np.array([lab[n] == 'ok' for n in H], int); w = np.array([items[n]['weight'] for n in H]); K = [items[n]['key'] for n in H]
style = np.array([items[n]['style'] for n in H])
HS = {k: hadm_scores(k) for k in K}
with open(D / 'hadm/hadm_gold_scores.jsonl', 'w') as f:
    for n in items: f.write(json.dumps(dict(num=n, key=items[n]['key'], label=lab[n], style=items[n]['style'], **hadm_scores(items[n]['key']))) + '\n')
S = {f'HADM-L {r}': -np.array([HS[k][r] for k in K]) for r in HS[K[0]]}
S['gemma_p_ok'] = np.array([fu[k]['gemma_p_ok'] for k in K]); S['det_conf'] = np.array([fu[k]['det_conf'] for k in K])
S['teacher_ok (fused)'] = np.array([fu[k]['teacher_ok'] for k in K]); S['clip_l14_logreg [gate 0.95]'] = np.array([stu[k]['student_p_ok'] for k in K])
r01 = lambda x: rankdata(x) / len(x)
hb = 'HADM-L hand iou_hand>=0.1'
S[f'fusion* rank({hb}, det_conf)'] = r01(S[hb]) + r01(S['det_conf'])
S[f'fusion* rank({hb}, gemma, det_conf)'] = r01(S[hb]) + r01(S['gemma_p_ok']) + r01(S['det_conf'])
REF = ('gemma_p_ok', 'det_conf', 'teacher_ok (fused)', 'clip_l14_logreg [gate 0.95]')

def metrics(yy, x, ww):
    b = 1 - yy; out = dict(auc=roc_auc_score(b, -x, sample_weight=ww), ap_bad=average_precision_score(b, -x, sample_weight=ww))
    fpr, tpr, _ = roc_curve(b, -x, sample_weight=ww)
    for f in (0.05, 0.10): out[f'rec@{int(f * 100)}fpr'] = float(tpr[fpr <= f + 1e-12].max())
    return out
def boot_idx(sub, seed):
    rng = np.random.default_rng(seed); io, ib = sub[y[sub] == 1], sub[y[sub] == 0]
    return [np.concatenate([rng.choice(io, len(io)), rng.choice(ib, len(ib))]) for _ in range(a.boot)]
def with_ci(x, sub, BI):
    pt = metrics(y[sub], x[sub], w[sub]); bs = [metrics(y[ix], x[ix], w[ix]) for ix in BI]
    return {k: (float(pt[k]), float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))) for k in pt}
allix = np.arange(len(y)); BI = boot_idx(allix, a.seed)  # same construction + seed as 07 -> identical resamples
# paired bootstrap: P(AUC_hadm > AUC_ref) over the same resamples
def auc_on(x, ix): return roc_auc_score(1 - y[ix], -x[ix], sample_weight=w[ix])
res = dict(meta=dmeta, n_hands=len(H), n_bad=int((y == 0).sum()), n_boot=a.boot, scores={s: with_ci(x, allix, BI) for s, x in S.items()},
           paired={}, style={}, bad_rank_pct={})
f3 = f'fusion* rank({hb}, gemma, det_conf)'
for cand in (hb, f3):
    for ref in ('gemma_p_ok', 'det_conf', 'clip_l14_logreg [gate 0.95]'):
        d = np.array([auc_on(S[cand], ix) - auc_on(S[ref], ix) for ix in BI])
        res['paired'][f'{cand} − {ref}'] = dict(delta=float(auc_on(S[cand], allix) - auc_on(S[ref], allix)), ci=(float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))), p_better=float((d > 0).mean()))
for st in ('anime', 'art', 'photo'):
    sub = np.where(style == st)[0]; nb = int((y[sub] == 0).sum())
    if nb == 0: res['style'][st] = dict(n=len(sub), n_bad=0); continue
    B2 = boot_idx(sub, a.seed + 1)
    res['style'][st] = dict(n=len(sub), n_bad=nb, scores={s: with_ci(S[s], sub, B2)['auc'] for s in (hb, 'HADM-L hand iod_crop>=0.5', 'HADM-L hand whole_image') + REF})
bad = [n for n in H if lab[n] == 'bad']
res['bad_rank_pct'] = {s: {n: float((S[s] < S[s][H.index(n)]).mean()) for n in bad} for s in S}
res['bad_raw'] = {n: dict(style=items[n]['style'], **{k: round(v, 3) for k, v in HS[items[n]['key']].items() if k.startswith('hand')}) for n in bad}
res['ok_hand_score_pct'] = {r: [float(np.percentile([HS[k][r] for k, yy in zip(K, y) if yy == 1], q)) for q in (50, 90)] for r in ('hand iou_hand>=0.1', 'hand whole_image')}
json.dump(res, open(G / 'eval_hadm.json', 'w'), indent=1)
c3 = lambda t, d=3: f'{t[0]:.{d}f} [{t[1]:.{d - 1}f}–{t[2]:.{d - 1}f}]'
L = ['# HADM-L (Adobe human-artifact detector) as a hand-quality score on the gold check', '',
     f"Same {len(H)} gold hands ({res['n_bad']} BAD), weights and {a.boot} label-stratified bootstrap resamples (seed {a.seed}) as eval.md. "
     f"HADM-L ran on Kaggle ({dmeta['gpu']}, torch {dmeta['torch']}, {dmeta['sec_per_image']:.2f} s/image) on the full-resolution Civitai originals "
     "(long side 1024; 144/145 images, 1 local 512-px fallback), detections kept down to score 0.001. "
     "`hand` = only HADM's hand-artifact class, `any` = any of its 6 artifact classes. Score = max matched artifact score (0 if none), negated.", '',
     '`fusion*` rows are rank averages chosen after seeing the gold results: optimistic, for orientation only.', '',
     '| score | AUC | AP (BAD) | recall BAD @5% FPR | recall BAD @10% FPR |', '|---|---|---|---|---|']
for s, r in sorted(res['scores'].items(), key=lambda t: -t[1]['auc'][0]):
    L.append(f"| {('*ours:* ' if s in REF else '') + s} | {c3(r['auc'])} | {c3(r['ap_bad'])} | {c3(r['rec@5fpr'], 2)} | {c3(r['rec@10fpr'], 2)} |")
L += ['', f'AP of a random score ≈ BAD prevalence {1 - np.average(y, weights=w):.3f}.', '',
      '## Paired bootstrap: AUC(candidate) − AUC(reference) on the same resamples', '', '| candidate − reference | ΔAUC | 95% CI | P(candidate better) |', '|---|---|---|---|']
for ref, r in res['paired'].items(): L.append(f"| {ref} | {r['delta']:+.3f} | [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | {r['p_better']:.2f} |")
L += ['', '## By style (AUC [95% CI]; tiny BAD counts, read as anecdotes)', '', '| style | hands | BAD | ' + ' | '.join(
      (hb, 'HADM-L hand iod_crop>=0.5', 'HADM-L hand whole_image') + REF) + ' |', '|---' * (3 + 3 + len(REF)) + '|']
for st, r in res['style'].items():
    L.append(f"| {st} | {r['n']} | {r['n_bad']} | " + (' | '.join(c3(v) for v in r['scores'].values()) if 'scores' in r else '–') + ' |')
L += ['', '## Each BAD crop: raw HADM hand-artifact score per rule, and percentile rank among gold hands (0 = flagged first)', '',
      f"OK hands, median / 90th percentile of the raw score: iou_hand>=0.1 {res['ok_hand_score_pct']['hand iou_hand>=0.1'][0]:.2f} / {res['ok_hand_score_pct']['hand iou_hand>=0.1'][1]:.2f}; "
      f"whole_image {res['ok_hand_score_pct']['hand whole_image'][0]:.2f} / {res['ok_hand_score_pct']['hand whole_image'][1]:.2f}.", '',
      '| # | style | raw iou_hand>=0.1 | raw iod_crop>=0.5 | raw whole_image | rank HADM iou_hand>=0.1 | rank gemma | rank det_conf | rank CLIP-L student |', '|---|---|---|---|---|---|---|---|---|']
for n in bad:
    rr = res['bad_raw'][n]; pr = res['bad_rank_pct']
    L.append(f"| {n} | {rr['style']} | {rr['hand iou_hand>=0.1']:.3f} | {rr['hand iod_crop>=0.5']:.3f} | {rr['hand whole_image']:.3f} | {pr[hb][n]:.2f} | {pr['gemma_p_ok'][n]:.2f} | {pr['det_conf'][n]:.2f} | {pr['clip_l14_logreg [gate 0.95]'][n]:.2f} |")
open(G / 'eval_hadm.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
