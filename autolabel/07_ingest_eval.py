"""Step 7: ingest the user's gold-check answers -> calibration + honest evaluation with bootstrap CIs.

Answers file (default autolabel_data/gold_check/answers.txt), free-form numbers after each label, e.g.
    bad: 4, 17 33
    nohand: 2 40
    skip: 9
Anything not listed is an OK hand. Numbers are the global numbers printed on the contact sheets.

What it computes (every metric re-weighted with the stratified-sampling weights from manifest.json):
* OK-vs-BAD on the gold hands (labels ok/bad; nohand and skip are left out), for every score: the fused teacher,
  Gemma alone, CLIP-B/32 zero-shot, detector confidence and every student (current `student_*.jsonl` and, if present,
  the pre-gate-fix students in `v1_gate050/`). BAD is the positive class:
  AUC, AP(BAD), recall of BAD at 5% / 10% FPR (read off the weighted ROC of the gold hands), ECE (10 bins) and Brier.
  95% CIs: label-stratified bootstrap (B resamples; the 9-ish BAD are resampled with replacement, so CIs are wide).
* Calibration for probability-like scores: Platt (logistic on logit(score)) and isotonic, evaluated leave-one-out
  (each gold hand gets a calibrated probability from a calibrator fitted on all the other hands). Platt is the deployed
  choice: 2 parameters; isotonic overfits with < 10 positives.
  models/calibration.json stores Platt (+ isotonic for reference) for the main score, fitted on all gold hands.
* Hand gate: agreement of Gemma p_hand (several thresholds) and detector confidence with the user's 'nohand' answers.
* Per-BAD-crop ranks under each score (failure analysis).
Writes autolabel_data/gold_check/eval.json, eval.md and models/calibration.json.
Usage: python autolabel/07_ingest_eval.py [--answers path] [--main clip_l14_logreg] [--boot 2000]
"""
import argparse, json, re
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from common import R, D, read_jsonl, ece
ap = argparse.ArgumentParser(); ap.add_argument('--gold-dir', default='gold_check'); ap.add_argument('--answers', default=None)
ap.add_argument('--main', default='clip_l14_logreg', help='student tag used for models/calibration.json')
ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
G = D / a.gold_dir; a.answers = a.answers or str(G / 'answers.txt')
man = json.load(open(G / 'manifest.json')); items = {it['num']: it for it in man['items']}
lab = {}
for line in open(a.answers):
    line = line.split('#')[0]; m = re.match(r'\s*(bad|nohand|skip)\s*:(.*)', line, re.I)
    if m:
        for n in map(int, re.findall(r'\d+', m.group(2))):
            if n not in items: raise SystemExit(f'number {n} not in gold set')
            if n in lab and lab[n] != m.group(1).lower(): raise SystemExit(f'number {n} has two labels')
            lab[n] = m.group(1).lower()
for n in items: lab.setdefault(n, 'ok')
fu = read_jsonl(D / 'fused.jsonl')
fmeta = json.load(open(D / 'fused_meta.json')); HAND_THR = fmeta.get('hand_thr', 0.5)
scores = {'teacher_ok (fused)': lambda k: fu[k]['teacher_ok'], 'gemma_p_ok': lambda k: fu[k]['gemma_p_ok'],
          'clip_b32_zs': lambda k: fu[k]['clip_p_ok'], 'det_conf': lambda k: fu[k]['det_conf']}
PROB = {'gemma_p_ok', 'clip_b32_zs'}
for p in sorted(D.glob('student_*.jsonl')) + sorted((D / 'v1_gate050').glob('student_*.jsonl')):
    if p.stem.endswith('_dry'): continue
    nm = p.stem[len('student_'):] + ('  [gate 0.50, v1]' if p.parent.name == 'v1_gate050' else f'  [gate {HAND_THR:.2f}]')
    st = read_jsonl(p); scores[nm] = (lambda st: (lambda k: st[k]['student_p_ok']))(st); PROB.add(nm)
main = next(s for s in scores if s.startswith(a.main + '  [gate') and 'v1' not in s)
H = [n for n in items if lab[n] in ('ok', 'bad')]
y = np.array([lab[n] == 'ok' for n in H], int); w = np.array([items[n]['weight'] for n in H]); K = [items[n]['key'] for n in H]
S = {s: np.array([f(k) for k in K], float) for s, f in scores.items()}
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))

def metrics(yy, x, ww, prob):
    """yy = 1 for OK; x = P(ok)-like score (higher = more OK). BAD is the positive class."""
    b = 1 - yy; out = dict(auc=roc_auc_score(b, -x, sample_weight=ww), ap_bad=average_precision_score(b, -x, sample_weight=ww))
    fpr, tpr, _ = roc_curve(b, -x, sample_weight=ww)
    for f in (0.05, 0.10): out[f'rec@{int(f * 100)}fpr'] = float(tpr[fpr <= f + 1e-12].max())
    if prob: out['ece'] = ece(yy, x, w=ww); out['brier'] = float(np.average((x - yy) ** 2, weights=ww))
    return out

def loo_cal(x, yy, ww):
    pl, iso = np.zeros(len(x)), np.zeros(len(x))
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        c = LogisticRegression(C=1e4).fit(logit(x[m])[:, None], yy[m], sample_weight=ww[m]); pl[i] = c.predict_proba(logit(x[i:i + 1])[:, None])[0, 1]
        iso[i] = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(x[m], yy[m], sample_weight=ww[m]).predict(x[i:i + 1])[0]
    return pl, iso

rng = np.random.default_rng(a.seed); io, ib = np.where(y == 1)[0], np.where(y == 0)[0]
BI = [np.concatenate([rng.choice(io, len(io)), rng.choice(ib, len(ib))]) for _ in range(a.boot)]
def with_ci(fn):
    pt = fn(np.arange(len(y))); bs = [fn(ix) for ix in BI]
    return {k: (float(pt[k]), float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))) for k in pt}
res = dict(main=main, hand_thr=HAND_THR, n_boot=a.boot, counts={c: sum(v == c for v in lab.values()) for c in ('ok', 'bad', 'nohand', 'skip')},
           bad_nums=sorted(n for n in H if lab[n] == 'bad'), scores={}, calibration={})
for s, x in S.items():
    res['scores'][s] = with_ci(lambda ix, x=x, s=s: metrics(y[ix], x[ix], w[ix], s in PROB))
for s in [s for s in S if s in PROB]:
    pl, iso = loo_cal(S[s], y, w)
    res['calibration'][s] = {nm: with_ci(lambda ix, p=p: dict(ece=ece(y[ix], p[ix], w=w[ix]), brier=float(np.average((p[ix] - y[ix]) ** 2, weights=w[ix]))))
                             for nm, p in (('raw', S[s]), ('platt_loo', pl), ('isotonic_loo', iso))}
base = np.full(len(y), np.average(y, weights=w))
res['calibration']['constant base rate (reference)'] = {'raw': with_ci(lambda ix: dict(ece=ece(y[ix], base[ix], w=w[ix]), brier=float(np.average((base[ix] - y[ix]) ** 2, weights=w[ix]))))}
# deployed calibrator: Platt on all gold hands (isotonic stored for reference only)
x = S[main]; pl = LogisticRegression(C=1e4).fit(logit(x)[:, None], y, sample_weight=w)
iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(x, y, sample_weight=w)
(R / 'models').mkdir(exist_ok=True)
json.dump(dict(score=main.split('  ')[0], model_file=f"models/hand_ok_{main.split('  ')[0]}.onnx", method='platt',
               platt=dict(a=float(pl.coef_[0, 0]), b=float(pl.intercept_[0]), formula='p_ok = sigmoid(a*logit(student_p_ok)+b)'),
               isotonic_reference_only=dict(x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist()),
               n_gold_hands=len(H), n_bad=int((y == 0).sum()), hand_thr=HAND_THR,
               note='fitted on the 150-crop user gold check (9 BAD); treat as approximate'),
          open((R / 'models/calibration.json') if a.gold_dir == 'gold_check' else (G / 'calibration.json'), 'w'), indent=1)
# ---- hand gate vs the user's nohand answers (skip excluded)
gi = [n for n in items if lab[n] != 'skip']; tn = np.array([lab[n] == 'nohand' for n in gi]); gw = np.array([items[n]['weight'] for n in gi])
ph = np.array([fu[items[n]['key']]['gemma_p_hand'] for n in gi]); dc = np.array([fu[items[n]['key']]['det_conf'] for n in gi])
gl = np.array([lab[n] for n in gi])
gate = dict(n=len(gi), n_nohand=int(tn.sum()), weighted_nohand_rate=float(np.average(tn, weights=gw)),
            auc_p_hand=float(roc_auc_score(tn, -ph, sample_weight=gw)), auc_det_conf=float(roc_auc_score(tn, -dc, sample_weight=gw)), rules={})
def rule(pred, nm):
    tp, fp, fn = int((pred & tn).sum()), int((pred & ~tn).sum()), int((~pred & tn).sum())
    gate['rules'][nm] = dict(tp=tp, fp=fp, fn=fn, precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn),
                             nohand_left_in_hands=fn, bad_lost=int((pred & (gl == 'bad')).sum()), ok_lost=int((pred & (gl == 'ok')).sum()),
                             nohand_share_of_passed=float(np.average(tn[~pred], weights=gw[~pred])))
for t in (0.5, 0.8, 0.9, 0.95, 0.98): rule(ph < t, f'p_hand < {t}')
for t in (0.3, 0.5): rule((ph < 0.95) | (dc < t), f'p_hand < 0.95 or det_conf < {t}')
res['gate'] = gate
# ---- per-BAD ranks: percentile of each BAD crop among gold hands (0 = most suspicious)
pct = {s: {n: float((S[s] < S[s][H.index(n)]).mean()) for n in res['bad_nums']} for s in S}
res['bad_rank_pct'] = pct
json.dump(res, open(G / 'eval.json', 'w'), indent=1)
# ---- markdown
c3 = lambda t, d=3: f'{t[0]:.{d}f} [{t[1]:.{d - 1}f}–{t[2]:.{d - 1}f}]'
L = [f"# Gold-check evaluation ({len(items)} crops judged by the user)", '',
     f"labels: {res['counts']} → **{len(H)} gold hands, of which {int((y == 0).sum())} BAD**. Point estimate [95% bootstrap CI, "
     f"{a.boot} label-stratified resamples]. With only {int((y == 0).sum())} BAD the CIs are wide: one BAD crop moves recall by ~11 points.", '',
     f"Current hand gate: Gemma p_hand >= {HAND_THR}. Main score (calibration.json): `{main}`.", '',
     '## OK vs BAD (BAD = positive class)', '', '| score | AUC | AP (BAD) | recall BAD @5% FPR | recall BAD @10% FPR |', '|---|---|---|---|---|']
for s, r in sorted(res['scores'].items(), key=lambda t: -t[1]['auc'][0]):
    L.append(f"| {s} | {c3(r['auc'])} | {c3(r['ap_bad'])} | {c3(r['rec@5fpr'], 2)} | {c3(r['rec@10fpr'], 2)} |")
L += ['', f"AP of a random score = BAD prevalence ≈ {1 - np.average(y, weights=w):.3f}.", '',
      '## Calibration (ECE 10 bins / Brier; Platt and isotonic evaluated leave-one-out)', '', '| score | ECE raw | ECE Platt | ECE isotonic | Brier raw | Brier Platt | Brier isotonic |', '|---|---|---|---|---|---|---|']
for s, r in res['calibration'].items():
    g = lambda nm, k: c3(r[nm][k]) if nm in r else '–'
    L.append(f"| {s} | {g('raw', 'ece')} | {g('platt_loo', 'ece')} | {g('isotonic_loo', 'ece')} | {g('raw', 'brier')} | {g('platt_loo', 'brier')} | {g('isotonic_loo', 'brier')} |")
L += ['', '## Hand gate vs the user\'s "nohand" answers', '', f"{gate['n']} non-skip crops, {gate['n_nohand']} nohand "
      f"(weighted rate {gate['weighted_nohand_rate']:.1%}). AUC for nohand: Gemma p_hand {gate['auc_p_hand']:.3f}, detector conf {gate['auc_det_conf']:.3f}.", '',
      '| rule → "not a hand" | precision | recall | nohand left among hands | OK hands lost | BAD hands lost | nohand share of passed crops |', '|---|---|---|---|---|---|---|']
for nm, r in gate['rules'].items():
    L.append(f"| {nm} | {r['precision']:.2f} | {r['recall']:.2f} | {r['nohand_left_in_hands']} | {r['ok_lost']} | {r['bad_lost']} | {r['nohand_share_of_passed']:.1%} |")
L += ['', '## Rank of each BAD crop (percentile among gold hands; 0 = most suspicious, flagged first)', '',
      '| # | ' + ' | '.join(s.split('  ')[0] + (' v1' if 'v1' in s else '') for s in S) + ' |', '|---' * (len(S) + 1) + '|']
for n in res['bad_nums']: L.append(f'| {n} | ' + ' | '.join(f'{pct[s][n]:.2f}' for s in S) + ' |')
open(G / 'eval.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
