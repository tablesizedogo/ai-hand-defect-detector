"""Step 8: evaluate ready-made external hand-quality models on the user's gold check (same protocol as step 7).

Models (weights downloaded to weights/external/, git-ignored):
* CountHallu RealHand quality classifier (arXiv:2510.13080), https://huggingface.co/ShyFoo/CountHallu-quality_cls_model-RealHand
  timm `maxvit_small_tf_224`, 2 classes, index 1 = clean/countable, index 0 = failed. Apache-2.0.
  Preprocessing from the model card: Resize(224), ToTensor, ImageNet mean/std. Score = softmax[:, 1] = P(clean).
* CountHallu RealHand finger counter, https://huggingface.co/ShyFoo/CountHallu-counting_model-RealHand
  Ultralytics YOLO (v13), 1 class 'fingertip'; a correct hand has 5. Paper settings imgsz=640, conf=0.25, iou=0.1. AGPL-3.0.
  Scores (higher = more OK):
    fingers_hard  = -|n - 5|, n = #boxes at conf 0.25 (the paper's rule: bad iff n != 5)
    fingers_soft  = P(n == 5) with n ~ Poisson-binomial over all boxes at conf >= 0.05 (each box = Bernoulli(conf))
* countHallu pipeline: P(clean) * P(n == 5) (quality gate, then counting, as in the paper's two-stage protocol).

Crop context variants: `p13` = the original gold crop files (square, 1.3x box, from the 512-px images, as every other score saw);
`p20` = re-cropped from the same 512-px source images with 2.0x padding (more arm/context).

Metrics, eval subset (gold OK + BAD hands; nohand/skip excluded), stratified-sampling weights and the label-stratified bootstrap
(same seed and resample construction) are identical to 07_ingest_eval.py, so the numbers compare 1:1 with eval.md.
Writes autolabel_data/gold_check/external_scores.jsonl, eval_external.json, eval_external.md.
Usage: OMP_NUM_THREADS=2 nice -n 15 ../.venv/bin/python 08_eval_external.py [--boot 2000] [--threads 2]
"""
import argparse, json, re
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from common import R, D, ece
ap = argparse.ArgumentParser(); ap.add_argument('--gold-dir', default='gold_check'); ap.add_argument('--answers', default=None)
ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--threads', type=int, default=2); ap.add_argument('--rescore', action='store_true'); a = ap.parse_args()
G = D / a.gold_dir; a.answers = a.answers or str(G / 'answers.txt'); W = R / 'weights/external'
man = json.load(open(G / 'manifest.json')); items = {it['num']: it for it in man['items']}
lab = {}
for line in open(a.answers):
    line = line.split('#')[0]; m = re.match(r'\s*(bad|nohand|skip)\s*:(.*)', line, re.I)
    if m:
        for n in map(int, re.findall(r'\d+', m.group(2))): lab[n] = m.group(1).lower()
for n in items: lab.setdefault(n, 'ok')
meta = {json.loads(l)['key']: json.loads(l) for l in open(D / 'manifest.jsonl')}
pre = {json.loads(l)['id']: json.loads(l) for l in open(R / 'data/preannotations.jsonl')}

def crop(key, pad):
    if pad == 1.3: return Image.open(D / f'crops/{key}.jpg').convert('RGB')
    m = meta[key]; im = Image.open(Path('/workspace/rag_civitai') / pre[m['id']]['path']).convert('RGB')
    x0, y0, x1, y1 = m['xyxy']; s = max(x1 - x0, y1 - y0) * pad / 2; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    c = im.crop((int(max(0, cx - s)), int(max(0, cy - s)), int(min(im.width, cx + s)), int(min(im.height, cy + s)))); c.thumbnail((448, 448)); return c

def p_exact(confs, k=5):
    """Poisson-binomial P(sum == k) for independent Bernoulli(conf)."""
    d = np.zeros(len(confs) + 2); d[0] = 1.0
    for p in confs: d[1:] = d[1:] * (1 - p) + d[:-1] * p; d[0] *= (1 - p)
    return float(d[k]) if k < len(d) else 0.0


out_p = G / 'external_scores.jsonl'; VARIANTS = {'p13': 1.3, 'p20': 2.0}
if a.rescore or not out_p.exists():
    import torch, timm
    from torchvision import transforms as T
    torch.set_num_threads(a.threads)
    q = timm.create_model('maxvit_small_tf_224', pretrained=False, num_classes=2)
    q.load_state_dict(torch.load(W / 'maxvit_quality.pth', map_location='cpu')); q.eval()
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    from ultralytics import YOLO
    yolo = YOLO(str(W / 'yolo13_fingers.pt'), task='detect')
    with open(out_p, 'w') as f:
        for n, it in sorted(items.items()):
            r = dict(num=n, key=it['key'], label=lab[n])
            for vn, pad in VARIANTS.items():
                im = crop(it['key'], pad)
                with torch.no_grad(): r[f'q_clean_{vn}'] = float(torch.softmax(q(tf(im)[None]), 1)[0, 1])
                res = yolo.predict(source=im, imgsz=640, conf=0.05, iou=0.1, verbose=False, device='cpu')[0]
                cf = res.boxes.conf.numpy().tolist() if len(res.boxes) else []
                r[f'n_fingers_{vn}'] = int(sum(c >= 0.25 for c in cf)); r[f'p5_{vn}'] = p_exact(cf); r[f'confs_{vn}'] = [round(c, 3) for c in cf]
            f.write(json.dumps(r) + '\n'); f.flush(); print(n, lab[n], {k: v for k, v in r.items() if not k.startswith('confs')}, flush=True)
X = {json.loads(l)['num']: json.loads(l) for l in open(out_p)}
scores = {}
for vn in VARIANTS:
    scores[f'countHallu quality P(clean) [{vn}]'] = lambda n, vn=vn: X[n][f'q_clean_{vn}']
    scores[f'countHallu fingers hard -|n-5| [{vn}]'] = lambda n, vn=vn: -abs(X[n][f'n_fingers_{vn}'] - 5)
    scores[f'countHallu fingers soft P(n=5) [{vn}]'] = lambda n, vn=vn: X[n][f'p5_{vn}']
    scores[f'countHallu pipeline P(clean)*P(n=5) [{vn}]'] = lambda n, vn=vn: X[n][f'q_clean_{vn}'] * X[n][f'p5_{vn}']
PROB = {s for s in scores if 'P(' in s}
H = [n for n in items if lab[n] in ('ok', 'bad')]
y = np.array([lab[n] == 'ok' for n in H], int); w = np.array([items[n]['weight'] for n in H])
S = {s: np.array([f(n) for n in H], float) for s, f in scores.items()}

def metrics(yy, x, ww, prob):
    b = 1 - yy; out = dict(auc=roc_auc_score(b, -x, sample_weight=ww), ap_bad=average_precision_score(b, -x, sample_weight=ww))
    fpr, tpr, _ = roc_curve(b, -x, sample_weight=ww)
    for f in (0.05, 0.10): out[f'rec@{int(f * 100)}fpr'] = float(tpr[fpr <= f + 1e-12].max())
    if prob: out['ece'] = ece(yy, x, w=ww); out['brier'] = float(np.average((x - yy) ** 2, weights=ww))
    return out
rng = np.random.default_rng(a.seed); io, ib = np.where(y == 1)[0], np.where(y == 0)[0]
BI = [np.concatenate([rng.choice(io, len(io)), rng.choice(ib, len(ib))]) for _ in range(a.boot)]
def with_ci(fn):
    pt = fn(np.arange(len(y))); bs = [fn(ix) for ix in BI]
    return {k: (float(pt[k]), float(np.percentile([b[k] for b in bs], 2.5)), float(np.percentile([b[k] for b in bs], 97.5))) for k in pt}
res = dict(n_hands=len(H), n_bad=int((y == 0).sum()), n_boot=a.boot, scores={}, rules={}, bad_rank_pct={})
for s, x in S.items(): res['scores'][s] = with_ci(lambda ix, x=x, s=s: metrics(y[ix], x[ix], w[ix], s in PROB))
# hard decision rules of the paper (weighted FPR on OK, recall on BAD)
def rule(pred_bad, nm):
    res['rules'][nm] = dict(recall_bad=float(pred_bad[y == 0].mean()), fpr_weighted=float(np.average(pred_bad[y == 1], weights=w[y == 1])),
                            n_flagged_ok=int(pred_bad[y == 1].sum()), n_flagged_bad=int(pred_bad[y == 0].sum()))
for vn in VARIANTS:
    qc = np.array([X[n][f'q_clean_{vn}'] for n in H]); nf = np.array([X[n][f'n_fingers_{vn}'] for n in H])
    rule(qc < 0.5, f'quality P(clean) < 0.5 [{vn}]'); rule(nf != 5, f'fingers n != 5 [{vn}]'); rule((qc < 0.5) | (nf != 5), f'paper pipeline: failed or n != 5 [{vn}]')
bad = [n for n in H if lab[n] == 'bad']
res['bad_rank_pct'] = {s: {n: float((S[s] < S[s][H.index(n)]).mean()) for n in bad} for s in S}
res['finger_count_hist'] = {vn: {c: {str(k): int(sum(X[n][f'n_fingers_{vn}'] == k for n in H if lab[n] == c)) for k in range(0, 11)} for c in ('ok', 'bad')} for vn in VARIANTS}
json.dump(res, open(G / 'eval_external.json', 'w'), indent=1)
ours = json.load(open(G / 'eval.json'))['scores']
c3 = lambda t, d=3: f'{t[0]:.{d}f} [{t[1]:.{d - 1}f}–{t[2]:.{d - 1}f}]'
L = ['# External baselines on the gold check', '', f"Same {len(H)} gold hands ({res['n_bad']} BAD), weights and {a.boot} bootstrap resamples as eval.md. "
     "`p13` = original gold crops (1.3x box), `p20` = 2.0x padding from the same 512-px images.", '',
     '| score | AUC | AP (BAD) | recall BAD @5% FPR | recall BAD @10% FPR |', '|---|---|---|---|---|']
rows = [(s, r) for s, r in res['scores'].items()] + [(f'*ours:* {s}', ours[s]) for s in ('det_conf', 'gemma_p_ok', 'teacher_ok (fused)', 'clip_l14_logreg  [gate 0.95]') if s in ours]
for s, r in sorted(rows, key=lambda t: -t[1]['auc'][0]):
    L.append(f"| {s} | {c3(r['auc'])} | {c3(r['ap_bad'])} | {c3(r['rec@5fpr'], 2)} | {c3(r['rec@10fpr'], 2)} |")
L += ['', '## Hard decision rules (as used in the CountHallu paper)', '', '| rule → BAD | recall BAD | FPR (weighted) | OK flagged | BAD flagged |', '|---|---|---|---|---|']
for nm, r in res['rules'].items(): L.append(f"| {nm} | {r['recall_bad']:.2f} | {r['fpr_weighted']:.2f} | {r['n_flagged_ok']} | {r['n_flagged_bad']}/{res['n_bad']} |")
L += ['', '## Finger count at conf 0.25 (gold hands, unweighted counts)', '', '| variant | class | ' + ' | '.join(map(str, range(11))) + ' |', '|---' * 13 + '|']
for vn, d in res['finger_count_hist'].items():
    for c in ('ok', 'bad'): L.append(f'| {vn} | {c} | ' + ' | '.join(str(d[c][str(k)]) for k in range(11)) + ' |')
L += ['', '## Rank of each BAD crop (percentile among gold hands; 0 = flagged first)', '', '| # | ' + ' | '.join(s.replace('countHallu ', '') for s in S) + ' |', '|---' * (len(S) + 1) + '|']
for n in bad: L.append(f'| {n} | ' + ' | '.join(f"{res['bad_rank_pct'][s][n]:.2f}" for s in S) + ' |')
open(G / 'eval_external.md', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
