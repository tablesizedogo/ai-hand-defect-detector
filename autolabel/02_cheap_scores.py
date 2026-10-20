"""Step 2: cheap per-crop signals + embeddings (CPU).
Outputs in autolabel_data/:
  keys.json                       row order for every .npy below (= manifest order)
  clip_b32_zs.jsonl               CLIP ViT-B/32 zero-shot: p_ok (good vs bad hand prompts), p_nohand
  emb_clip_b32.npy  (N,512)       L2-normalised image embeddings
  emb_dinov2_base.npy (N,768)     DINOv2-base CLS embeddings (L2-normalised)
  emb_clip_l14.npy  (N,768)       CLIP ViT-L/14 (laion2b_s32b_b82k) embeddings
Usage: python 02_cheap_scores.py [b32|dino|l14 ...]   (default: all, in that order; finished parts are skipped)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
R = Path(__file__).resolve().parents[1]; D = R / 'autolabel_data'
torch.set_num_threads(int(os.environ.get('EMB_THREADS', '2')))
keys = [json.loads(l)['key'] for l in open(D / 'manifest.jsonl')]
json.dump(keys, open(D / 'keys.json', 'w'))
load = lambda k: Image.open(D / f'crops/{k}.jpg').convert('RGB')
def log(m): print(time.strftime('%H:%M:%S'), m, flush=True)

GOOD = ["a photo of a normal human hand with five fingers", "a well-drawn hand with correct anatomy", "a natural looking hand"]
BAD = ["a deformed hand with extra fingers", "a malformed mutated hand with fused fingers", "a distorted melted hand, bad anatomy"]
NOH = ["an image with no hands", "an animal paw", "a foot"]

def openclip_embed(arch, pretrained, out, zs=False, bs=32):
    import open_clip
    if (D / out).exists(): log(f'skip {out}'); return
    m, _, pre = open_clip.create_model_and_transforms(arch, pretrained=pretrained); m.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(keys), bs):
            x = torch.stack([pre(load(k)) for k in keys[i:i + bs]]); f = m.encode_image(x); feats.append(f / f.norm(dim=-1, keepdim=True))
            if i % (bs * 20) == 0: log(f'{out} {i}/{len(keys)}')
        F = torch.cat(feats)
        if zs:
            tok = open_clip.get_tokenizer(arch); T = m.encode_text(tok(GOOD + BAD + NOH)); T = T / T.norm(dim=-1, keepdim=True)
            L = 100 * F @ T.T; g, b, n = L[:, 0:3].logsumexp(1), L[:, 3:6].logsumexp(1), L[:, 6:9].logsumexp(1)
            p_ok = torch.sigmoid(g - b); p_noh = torch.softmax(torch.stack([g, b, n], 1), 1)[:, 2]
            with open(D / 'clip_b32_zs.jsonl', 'w') as f:
                for k, a, c in zip(keys, p_ok.tolist(), p_noh.tolist()): f.write(json.dumps(dict(key=k, p_ok=a, p_nohand=c)) + '\n')
    np.save(D / out, F.numpy().astype(np.float32)); log(f'saved {out} {tuple(F.shape)}')

def dino(out='emb_dinov2_base.npy', bs=32):
    if (D / out).exists(): log(f'skip {out}'); return
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained('facebook/dinov2-base'); m = AutoModel.from_pretrained('facebook/dinov2-base').eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(keys), bs):
            x = proc(images=[load(k) for k in keys[i:i + bs]], return_tensors='pt')
            f = m(**x).last_hidden_state[:, 0]; feats.append(f / f.norm(dim=-1, keepdim=True))
            if i % (bs * 20) == 0: log(f'{out} {i}/{len(keys)}')
    F = torch.cat(feats); np.save(D / out, F.numpy().astype(np.float32)); log(f'saved {out} {tuple(F.shape)}')

parts = sys.argv[1:] or ['b32', 'dino', 'l14']
for p in parts:
    if p == 'b32': openclip_embed('ViT-B-32', 'laion2b_s34b_b79k', 'emb_clip_b32.npy', zs=True)
    elif p == 'dino': dino()
    elif p == 'l14': openclip_embed('ViT-L-14', 'laion2b_s32b_b82k', 'emb_clip_l14.npy')
log('done')
