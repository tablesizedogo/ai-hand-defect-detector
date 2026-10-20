"""Step 4: fuse per-crop signals into one teacher score + "is a hand" gate.

teacher_ok = mean of rank-normalised [Gemma p_ok, detector confidence, CLIP-B/32 zero-shot p_ok]
             (equal weights: rank-avg of these three gave AUC 0.93 / 0.89 on the 40-crop study, see AUTO_LABEL_FEASIBILITY.md;
              a logistic fit on 35 hands would just overfit). Crops without a VLM answer yet use the mean of the available ranks
              and are flagged vlm_done=False.
is_hand    = Gemma p_hand >= HAND_THR (normalised P(None) < 0.5). In the study: non-hands 0.89-1.0 P(None), hands <= 0.04
             (one mushy hand at 0.95). Without a VLM answer is_hand=None (unknown).
Output: autolabel_data/fused.jsonl
"""
import json, sys
import numpy as np
from common import D, manifest, read_jsonl, rank01
HAND_THR = 0.5
M = manifest(); keys = [m['key'] for m in M]
vlm = read_jsonl(D / 'vlm_gemma4e4b.jsonl'); zs = read_jsonl(D / 'clip_b32_zs.jsonl')
g = np.array([vlm[k]['p_ok'] if k in vlm else np.nan for k in keys])
c = np.array([zs[k]['p_ok'] if k in zs else np.nan for k in keys])
d = np.array([m['det_conf'] for m in M], float)
ranks = np.vstack([rank01(g), rank01(d), rank01(c)])
fused = np.nanmean(ranks, 0)
with open(D / 'fused.jsonl', 'w') as f:
    for i, m in enumerate(M):
        k = m['key']; v = vlm.get(k)
        f.write(json.dumps(dict(key=k, style=m['style'], family=m['family'], det_conf=m['det_conf'],
            clip_p_ok=None if np.isnan(c[i]) else float(c[i]), gemma_p_ok=None if v is None else v['p_ok'],
            gemma_p_hand=None if v is None else v['p_hand'], vlm_done=v is not None,
            is_hand=None if v is None else bool(v['p_hand'] >= HAND_THR), teacher_ok=float(fused[i]))) + '\n')
nv = int((~np.isnan(g)).sum()); nh = sum(1 for k in keys if k in vlm and vlm[k]['p_hand'] >= HAND_THR)
print(f'fused {len(keys)} crops; VLM done {nv}; is_hand among VLM-done: {nh}/{nv}')
