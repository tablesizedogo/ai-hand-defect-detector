"""Step 12: Platt calibration of the raw HADM-L hand-artifact score -> P(bad), fitted on the pooled gold set (v1 + v2).
Raw score = max HADM `hand` detection score with >= 50% of its area inside the 1.3x square hand crop (0 if none), i.e. the
`hadm_hand` field of fused.jsonl. Model: P(bad) = sigmoid(a * raw + b), weighted by the (per-round rescaled) sampling weights.
Leave-one-out ECE/Brier are reported. Output: models/hadm_calibration.json (used by the scorer and the ComfyUI node)."""
import json, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import R, D, read_jsonl, ece
fu = read_jsonl(D / 'fused.jsonl'); X, Y, W = [], [], []
for gd in ('gold_check', 'gold_check_v2'):
    its = {it['num']: it for it in json.load(open(D / gd / 'manifest.json'))['items']}; lab = {}
    for line in open(D / gd / 'answers.txt'):
        m = re.match(r'\s*(bad|nohand|skip)\s*:(.*)', line.split('#')[0], re.I)
        if m:
            for n in map(int, re.findall(r'\d+', m.group(2))): lab[n] = m.group(1).lower()
    sc = len(its) / sum(it['weight'] for it in its.values())
    for n, it in its.items():
        l = lab.get(n, 'ok')
        if l in ('ok', 'bad'): X.append(fu[it['key']]['hadm_hand']); Y.append(int(l == 'bad')); W.append(it['weight'] * sc)
X, Y, W = np.array(X)[:, None], np.array(Y), np.array(W)
fit = lambda m: LogisticRegression(C=1e4).fit(X[m], Y[m], sample_weight=W[m])
loo = np.array([fit(np.arange(len(Y)) != i).predict_proba(X[i:i + 1])[0, 1] for i in range(len(Y))])
c = fit(np.ones(len(Y), bool)); a, b = float(c.coef_[0, 0]), float(c.intercept_[0])
raw_ece = ece(1 - Y, 1 - X[:, 0], w=W)
out = dict(score='hadm_hand_raw', formula='p_bad = sigmoid(a * raw + b)', a=a, b=b, n_hands=int(len(Y)), n_bad=int(Y.sum()),
           ece_raw_as_prob=float(raw_ece), ece_loo=float(ece(1 - Y, 1 - loo, w=W)), brier_loo=float(np.average((loo - Y) ** 2, weights=W)),
           base_rate_weighted=float(np.average(Y, weights=W)), gold='gold_check + gold_check_v2 (pooled, per-round rescaled weights)',
           example={str(r): float(1 / (1 + np.exp(-(a * r + b)))) for r in (0.0, 0.3, 0.5, 0.7, 0.9)},
           note='P(bad) among gate-passed hand crops of the Civitai corpus; 17 BAD only, treat as approximate')
json.dump(out, open(R / 'models/hadm_calibration.json', 'w'), indent=1); print(json.dumps(out, indent=1))
