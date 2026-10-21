"""Step 4: fuse per-crop signals into one teacher score + "is a hand" gate.

teacher_ok = mean of rank-normalised [Gemma p_ok, detector confidence, CLIP-B/32 zero-shot p_ok]
             (equal weights: rank-avg of these three gave AUC 0.93 / 0.89 on the 40-crop study, see AUTO_LABEL_FEASIBILITY.md;
              a logistic fit on 35 hands would just overfit). Crops without a VLM answer yet use the mean of the available ranks
              and are flagged vlm_done=False.
is_hand    = Gemma p_hand >= HAND_THR (default 0.95 since the gold check; originally 0.5, i.e. normalised P(None) < 0.5). In the study: non-hands 0.89-1.0 P(None), hands <= 0.04
             (one mushy hand at 0.95). Without a VLM answer is_hand=None (unknown).
Output: autolabel_data/fused.jsonl
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from common import D, manifest, read_jsonl, rank01
ap = argparse.ArgumentParser()
ap.add_argument('--vlm', default=str(D / 'vlm_gemma4e4b.jsonl'), help='VLM results jsonl (use a frozen snapshot for a reproducible partial run)')
ap.add_argument('--scored-only', action='store_true', help='rank-normalise only over crops that have a VLM answer; '
                'crops without one get teacher_ok=None (they are left out of silver/gold until the VLM covers them)')
ap.add_argument('--hand-thr', type=float, default=0.95, help='is_hand = Gemma p_hand >= this. Was 0.5 until the gold check: '
                '9 of the 12 gold crops the user called "not a hand" but the 0.5 gate passed had p_hand 0.71-0.92, while 95 of 97 OK '
                'and 8 of 8 BAD gate-passed hands had p_hand >= 0.95 (see gold_check/eval.md)')
a = ap.parse_args()
HAND_THR = a.hand_thr
M = manifest(); keys = [m['key'] for m in M]
vlm = read_jsonl(a.vlm); zs = read_jsonl(D / 'clip_b32_zs.jsonl')
g = np.array([vlm[k]['p_ok'] if k in vlm else np.nan for k in keys])
c = np.array([zs[k]['p_ok'] if k in zs else np.nan for k in keys])
d = np.array([m['det_conf'] for m in M], float)
if a.scored_only:
    sm = ~np.isnan(g); d = np.where(sm, d, np.nan); c = np.where(sm, c, np.nan)
ranks = np.vstack([rank01(g), rank01(d), rank01(c)])
with np.errstate(all='ignore'): fused = np.nanmean(ranks, 0)
with open(D / 'fused.jsonl', 'w') as f:
    for i, m in enumerate(M):
        k = m['key']; v = vlm.get(k)
        f.write(json.dumps(dict(key=k, style=m['style'], family=m['family'], det_conf=m['det_conf'],
            clip_p_ok=None if np.isnan(c[i]) else float(c[i]), gemma_p_ok=None if v is None else v['p_ok'],
            gemma_p_hand=None if v is None else v['p_hand'], vlm_done=v is not None,
            is_hand=None if v is None else bool(v['p_hand'] >= HAND_THR), teacher_ok=None if np.isnan(fused[i]) else float(fused[i]))) + '\n')
nv = int((~np.isnan(g)).sum()); nh = sum(1 for k in keys if k in vlm and vlm[k]['p_hand'] >= HAND_THR)
json.dump(dict(vlm_file=Path(a.vlm).name, vlm_sha256=hashlib.sha256(open(a.vlm, 'rb').read()).hexdigest(), scored_only=a.scored_only, hand_thr=HAND_THR,
               n_crops=len(keys), n_vlm=nv, n_is_hand=nh), open(D / 'fused_meta.json', 'w'), indent=1)
print(f'fused {len(keys)} crops; VLM done {nv}; is_hand among VLM-done: {nh}/{nv}')
