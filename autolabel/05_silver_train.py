"""Step 5: silver labels from confident tails of the teacher score + student classifier on frozen embeddings.

* Uses only crops with is_hand == True (VLM gate). --allow-partial lets it run before the VLM job finishes
  (then only VLM-done crops are used).
* Silver BAD = bottom --bad-q (default 15%) of teacher_ok, silver OK = top --ok-q (default 50%); the middle is discarded.
* Crops in the gold-check set (autolabel_data/gold_check/manifest.json) and the 40 study crops are never used for training.
* Student: logistic regression (default) or small MLP on L2-normalised embeddings; 5-fold CV agreement with the silver labels;
  predictions for ALL crops -> autolabel_data/student_<emb>_<model>.jsonl; ONNX export -> models/hand_ok_<emb>_<model>.onnx
  (input: float32 [N, dim] embedding, output: probabilities [N, 2], column 1 = P(ok)).
Examples:
  python autolabel/05_silver_train.py --emb clip_l14 --model logreg
  python autolabel/05_silver_train.py --emb dinov2_base --model mlp --allow-partial
"""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from common import R, D, read_jsonl
ap = argparse.ArgumentParser()
ap.add_argument('--emb', default='clip_l14', choices=['clip_b32', 'clip_l14', 'dinov2_base'])
ap.add_argument('--model', default='logreg', choices=['logreg', 'mlp'])
ap.add_argument('--bad-q', type=float, default=0.15); ap.add_argument('--ok-q', type=float, default=0.50)
ap.add_argument('--allow-partial', action='store_true')
ap.add_argument('--dry-run', action='store_true', help='pipeline test before the VLM finishes: crops without a VLM answer count as hands; outputs get a _dry suffix')
a = ap.parse_args()
keys = json.load(open(D / 'keys.json')); X = np.load(D / f'emb_{a.emb}.npy'); assert len(X) == len(keys)
fu = read_jsonl(D / 'fused.jsonl')
n_vlm = sum(fu[k]['vlm_done'] for k in keys)
if n_vlm < len(keys) and not (a.allow_partial or a.dry_run):
    raise SystemExit(f'VLM job not finished ({n_vlm}/{len(keys)}); rerun 04_fuse.py when done or pass --allow-partial')
excl = set()
gc = D / 'gold_check/manifest.json'
if gc.exists(): excl |= {it['key'] for it in json.load(open(gc))['items']}
if (R / 'auto/gold40.json').exists(): excl |= set(json.load(open(R / 'auto/gold40.json')))
pool = np.array([(fu[k]['is_hand'] is True or (a.dry_run and fu[k]['is_hand'] is None)) and k not in excl for k in keys])
t = np.array([np.nan if fu[k]['teacher_ok'] is None else fu[k]['teacher_ok'] for k in keys], float)
pool &= ~np.isnan(t)
lo, hi = np.quantile(t[pool], [a.bad_q, 1 - a.ok_q])
sel = pool & ((t <= lo) | (t >= hi)); y = (t[sel] >= hi).astype(int)
print(f'pool {pool.sum()} hands (excluded {len(excl)} eval crops); silver: {int((y == 0).sum())} BAD (<= {lo:.3f}), {int(y.sum())} OK (>= {hi:.3f}), discarded {pool.sum() - sel.sum()}')
mk = (lambda: LogisticRegression(C=1.0, max_iter=3000, class_weight='balanced')) if a.model == 'logreg' else \
     (lambda: MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0))
cv = cross_val_predict(mk(), X[sel], y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
from sklearn.metrics import average_precision_score, brier_score_loss, roc_curve
from common import ece
fpr_, tpr_, _ = roc_curve(1 - y, 1 - cv)  # positive class = silver BAD
rec_at = {f: float(tpr_[fpr_ <= f].max()) for f in (0.05, 0.10)}
cvm = dict(cv_auc=float(roc_auc_score(y, cv)), cv_acc=float(((cv >= .5) == y).mean()), cv_ap_bad=float(average_precision_score(1 - y, 1 - cv)),
           cv_recall_bad_at_fpr5=rec_at[0.05], cv_recall_bad_at_fpr10=rec_at[0.10], cv_brier=float(brier_score_loss(y, cv)), cv_ece=float(ece(y, cv)))
print('5-fold CV vs SILVER labels (agreement with the teacher, not ground truth): ' + '  '.join(f'{k[3:]}={v:.3f}' for k, v in cvm.items()))
clf = mk().fit(X[sel], y); p = clf.predict_proba(X)[:, 1]
tag = f'{a.emb}_{a.model}' + ('_dry' if a.dry_run else '')
with open(D / f'student_{tag}.jsonl', 'w') as f:
    for k, pk in zip(keys, p): f.write(json.dumps(dict(key=k, student_p_ok=float(pk), silver=None)) + '\n')
# optional sanity check on the 40 study crops (my labels; NOT the honest metric)
g40 = R / 'auto/gold40.json'
if g40.exists():
    g = json.load(open(g40)); idx = {k: i for i, k in enumerate(keys)}
    kk = [k for k in g if g[k] != 'nohand' and k in idx and not np.isnan(t[idx[k]])]
    yy = np.array([g[k] == 'ok' for k in kk]); print(f'study-40 sanity AUC ({len(kk)} hands with a VLM score; my labels, NOT honest): student={roc_auc_score(yy, p[[idx[k] for k in kk]]):.3f} '
          f'teacher={roc_auc_score(yy, t[[idx[k] for k in kk]]):.3f}')
from skl2onnx import to_onnx
(R / 'models').mkdir(exist_ok=True); onx = to_onnx(clf, X[:1].astype(np.float32), options={id(clf): {'zipmap': False}})
op = R / f'models/hand_ok_{tag}.onnx'; op.write_bytes(onx.SerializeToString())
import onnxruntime as ort
s = ort.InferenceSession(str(op)); po = s.run(None, {s.get_inputs()[0].name: X[:64].astype(np.float32)})[1][:, 1]
print(f'ONNX saved {op.name} ({op.stat().st_size // 1024} KB); max |onnx - sklearn| = {np.abs(po - p[:64]).max():.2e}')
json.dump(dict(emb=a.emb, model=a.model, bad_q=a.bad_q, ok_q=a.ok_q, thr_bad=float(lo), thr_ok=float(hi), n_bad=int((y == 0).sum()),
               n_ok=int(y.sum()), n_vlm_done=int(n_vlm), **cvm), open(D / f'student_{tag}_meta.json', 'w'), indent=1)
