"""Step 7: ingest the user's gold-check answers -> calibration + honest evaluation.

Answers file (default autolabel_data/gold_check/answers.txt), free-form numbers after each label, e.g.
    bad: 4, 17 33
    nohand: 2 40
    skip: 9
Anything not listed is an OK hand. Numbers are the global numbers printed on the contact sheets.

* Hand items (OK/BAD) are split 50/50 (stratified by label, seed) into a calibration half and a test half; we also swap the
  halves and report both (2-fold) because n is small.
* Calibrators fitted on the calibration half: Platt (logistic on logit(score)) and isotonic.
* Test-half metrics (all re-weighted with the stratified-sampling weights from manifest.json):
  AUC (OK vs BAD), recall on BAD at a fixed false-alarm rate (FPR on OK hands = 5% / 10%, threshold chosen on the
  calibration half), ECE (10 bins) and Brier for raw/Platt/isotonic probabilities.
* Gate: precision/recall of "not a hand" (is_hand == False) vs the user's nohand answers.
Writes autolabel_data/gold_check/eval.json, eval.md and models/calibration.json (Platt a,b + isotonic table for the main score).
Usage: python autolabel/07_ingest_eval.py [--answers path] [--main student_clip_l14_logreg]
"""
import argparse, json, re
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from common import R, D, read_jsonl, ece
ap = argparse.ArgumentParser(); ap.add_argument('--gold-dir', default='gold_check'); ap.add_argument('--answers', default=None)
ap.add_argument('--main', default=None, help='student tag (e.g. clip_l14_logreg); default: best available student, else teacher_ok')
ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
G = D / a.gold_dir; a.answers = a.answers or str(G / 'answers.txt')
man = json.load(open(G / 'manifest.json')); items = {it['num']: it for it in man['items']}
lab = {}
for line in open(a.answers):
    line = line.split('#')[0]; m = re.match(r'\s*(bad|nohand|skip)\s*:(.*)', line, re.I)
    if m:
        for n in map(int, re.findall(r'\d+', m.group(2))):
            if n not in items: raise SystemExit(f'number {n} not in gold set')
            lab[n] = m.group(1).lower()
for n in items: lab.setdefault(n, 'ok')
fu = read_jsonl(D / 'fused.jsonl')
scores = {'teacher_ok': lambda k: fu[k]['teacher_ok'], 'gemma_p_ok': lambda k: fu[k]['gemma_p_ok'],
          'clip_b32_zs': lambda k: fu[k]['clip_p_ok'], 'det_conf': lambda k: fu[k]['det_conf']}
studs = sorted(D.glob('student_*.jsonl'))
for p in studs:
    st = read_jsonl(p); scores[p.stem] = (lambda st: (lambda k: st[k]['student_p_ok']))(st)
main = ('student_' + a.main) if a.main else (studs[0].stem if studs else 'teacher_ok')
if a.main is None and studs:  # prefer CLIP-L logreg if present
    main = next((s.stem for s in studs if 'clip_l14_logreg' in s.stem), studs[0].stem)
H = [n for n in items if lab[n] in ('ok', 'bad')]
y = np.array([lab[n] == 'ok' for n in H], int); w = np.array([items[n]['weight'] for n in H]); K = [items[n]['key'] for n in H]
res = dict(main=main, counts={c: sum(v == c for v in lab.values()) for c in ('ok', 'bad', 'nohand', 'skip')}, auc_all={})
for s, f in scores.items():
    v = np.array([f(k) if f(k) is not None else np.nan for k in K], float); m = ~np.isnan(v)
    if m.sum() > 5 and len(set(y[m])) == 2: res['auc_all'][s] = float(roc_auc_score(y[m], v[m], sample_weight=w[m]))
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
x = np.array([scores[main](k) for k in K], float)
rng = np.random.default_rng(a.seed); idx_ok, idx_bad = np.where(y == 1)[0], np.where(y == 0)[0]; rng.shuffle(idx_ok); rng.shuffle(idx_bad)
A = np.concatenate([idx_ok[::2], idx_bad[::2]]); B = np.concatenate([idx_ok[1::2], idx_bad[1::2]])
def thr_at_fpr(xc, yc, wc, fpr):
    # flag BAD when score < thr; FPR = weighted share of OK hands flagged
    cand = np.unique(xc); best = cand[0]
    for t in cand:
        if np.average(xc[yc == 1] < t, weights=wc[yc == 1]) <= fpr: best = t
    return best
folds = []
for cal, te in ((A, B), (B, A)):
    pl = LogisticRegression(C=1e4).fit(logit(x[cal])[:, None], y[cal], sample_weight=w[cal])
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(x[cal], y[cal], sample_weight=w[cal])
    pr = {'raw': x[te], 'platt': pl.predict_proba(logit(x[te])[:, None])[:, 1], 'isotonic': iso.predict(x[te])}
    f = dict(n_cal=len(cal), n_test=len(te), n_bad_test=int((y[te] == 0).sum()),
             auc=float(roc_auc_score(y[te], x[te], sample_weight=w[te])) if len(set(y[te])) == 2 else None)
    for fpr in (0.05, 0.10):
        t = thr_at_fpr(x[cal], y[cal], w[cal], fpr); flag = x[te] < t
        f[f'recall_bad@fpr{int(fpr * 100)}'] = float(np.average(flag[y[te] == 0], weights=w[te][y[te] == 0])) if (y[te] == 0).any() else None
        f[f'realised_fpr@fpr{int(fpr * 100)}'] = float(np.average(flag[y[te] == 1], weights=w[te][y[te] == 1]))
        f[f'thr@fpr{int(fpr * 100)}'] = float(t)
    for nm, p in pr.items():
        f[f'ece_{nm}'] = float(ece(y[te], p, w=w[te])); f[f'brier_{nm}'] = float(np.average((p - y[te]) ** 2, weights=w[te]))
    folds.append(f)
res['folds'] = folds
res['mean'] = {k: float(np.mean([f[k] for f in folds if f[k] is not None])) for k in folds[0] if all(isinstance(f[k], (int, float)) or f[k] is None for f in folds) and any(f[k] is not None for f in folds)}
# final calibrator on ALL gold hands, for deployment
pl = LogisticRegression(C=1e4).fit(logit(x)[:, None], y, sample_weight=w)
iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(x, y, sample_weight=w)
(R / 'models').mkdir(exist_ok=True)
json.dump(dict(score=main, platt=dict(a=float(pl.coef_[0, 0]), b=float(pl.intercept_[0]), formula='p = sigmoid(a*logit(score)+b)'),
               isotonic=dict(x=iso.X_thresholds_.tolist(), y=iso.y_thresholds_.tolist()), n=len(H)), open((R / 'models/calibration.json') if a.gold_dir == 'gold_check' else (G / 'calibration.json'), 'w'), indent=1)
gi = [n for n in items if lab[n] != 'skip']
pred_noh = np.array([fu[items[n]['key']]['is_hand'] is False for n in gi]); true_noh = np.array([lab[n] == 'nohand' for n in gi])
res['gate'] = dict(tp=int((pred_noh & true_noh).sum()), fp=int((pred_noh & ~true_noh).sum()), fn=int((~pred_noh & true_noh).sum()),
                   precision=float((pred_noh & true_noh).sum() / max(1, pred_noh.sum())), recall=float((pred_noh & true_noh).sum() / max(1, true_noh.sum())))
json.dump(res, open(G / 'eval.json', 'w'), indent=1)
L = [f"# Gold-check evaluation ({len(items)} crops judged by the user)", '', f"labels: {res['counts']}; main score: `{main}`", '',
     '| score | AUC (all gold hands, weighted) |', '|---|---:|'] + [f'| {s} | {v:.3f} |' for s, v in sorted(res['auc_all'].items(), key=lambda t: -t[1])]
L += ['', '2-fold (calibrate on one half, test on the other), mean of both folds:', '', '| metric | value |', '|---|---:|'] + \
     [f'| {k} | {v:.3f} |' for k, v in res['mean'].items() if not k.startswith(('n_', 'thr'))]
L += ['', f"Gate 'not a hand': precision {res['gate']['precision']:.2f}, recall {res['gate']['recall']:.2f} (tp {res['gate']['tp']}, fp {res['gate']['fp']}, fn {res['gate']['fn']})",
      '', 'Metrics are re-weighted by the stratified sampling weights; with ~150 crops, CIs are wide (AUC about +-0.05-0.1).']
open(G / 'eval.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
