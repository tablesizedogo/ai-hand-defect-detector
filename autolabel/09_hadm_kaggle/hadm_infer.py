"""HADM-L inference on a folder of JPEGs -> per-image detections JSON (class, score, box).

Runs the official HADM code (github.com/wangkaihong/HADM) without compiling detectron2's C++ ops (inference of the
ViTDet/EVA-02-L Cascade R-CNN only needs torchvision ROIAlign/NMS) and on a current PyTorch instead of the pinned
torch 1.12/cu116:
* xformers.ops.memory_efficient_attention is replaced by torch.nn.functional.scaled_dot_product_attention
  (same math, same default scale 1/sqrt(head_dim)); apex FusedLayerNorm is not used by this config.
* Pillow>=10 aliases (Image.LINEAR/ANTIALIAS/CUBIC) are restored before importing detectron2.
* wandb / mmcv are stubbed if missing (training logging and soft-NMS only; soft-NMS is off in this config).
* activation checkpointing (fairscale) is switched off (inference only).
* test_score_thresh is lowered from 0.05 to --score-thresh (default 0.001) in all three cascade stages, so every region
  keeps a (possibly small) score per artifact class; detections_per_img stays 100.
* EMA weights are loaded and applied exactly as `tools/lazyconfig_train_net.py --inference` does
  (train.model_ema.enabled=True, use_ema_weights_for_eval_only=True).
Preprocessing = the repo's DefaultInferencer: BGR read, ResizeShortestEdge(1024, max_size=1024).
"""
import argparse, json, os, sys, time, types
ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True); ap.add_argument('--weights', required=True)
ap.add_argument('--images', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--score-thresh', type=float, default=0.001); ap.add_argument('--device', default='cuda')
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--random-init', action='store_true')
ap.add_argument('--config', default='projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py')
a = ap.parse_args()
sys.path.insert(0, a.repo)
mz = os.path.join(a.repo, 'detectron2/model_zoo/configs')
if not os.path.exists(mz): os.symlink(os.path.join(a.repo, 'configs'), mz)
# torch>=2.6 defaults torch.load(weights_only=True); the HADM training checkpoint (trusted, from the authors) needs the old behaviour
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
import torch, torch.nn.functional as F
from PIL import Image as _I  # Pillow>=10 removed these aliases; old detectron2 uses them at import time
for _o, _n in [('LINEAR', 'BILINEAR'), ('ANTIALIAS', 'LANCZOS'), ('CUBIC', 'BICUBIC')]:
    if not hasattr(_I, _o): setattr(_I, _o, getattr(_I, _n))
try: import wandb  # imported by detectron2.utils.events (training logging only)
except ImportError: sys.modules['wandb'] = types.ModuleType('wandb')
try: import mmcv.ops  # only soft_nms is imported, and the HADM-L config does not enable soft-NMS
except ImportError:
    def _no_soft_nms(*a, **k): raise RuntimeError('soft_nms needs mmcv')
    _m = types.ModuleType('mmcv'); _m.ops = types.ModuleType('mmcv.ops'); _m.ops.soft_nms = _no_soft_nms
    sys.modules['mmcv'] = _m; sys.modules['mmcv.ops'] = _m.ops
import detectron2.modeling.backbone.vit as vit
vit.xops = types.SimpleNamespace(memory_efficient_attention=lambda q, k, v: F.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2))
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import ema
from detectron2.engine.defaults import DefaultInferencer
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
setup_logger(name='fvcore'); setup_logger()  # prints checkpoint missing/unexpected keys
from detectron2.data.datasets.human_artifact_dataset import LOCAL_HUMAN_ARTIFACT_CATEGORIES as CATS

cfg = LazyConfig.load(os.path.join(a.repo, a.config))
cfg.model.backbone.net.use_act_checkpoint = False
for p in cfg.model.roi_heads.box_predictors: p.test_score_thresh = a.score_thresh
cfg.train.device = a.device; cfg.train.model_ema.enabled = True; cfg.train.model_ema.device = a.device
cfg.train.model_ema.use_ema_weights_for_eval_only = True
classes = [c['class_name'] for c in sorted(CATS, key=lambda c: c['id'])]
t0 = time.time()
model = instantiate(cfg.model); model.to(a.device)
ema.may_build_model_ema(cfg, model)
if not a.random_init:
    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(a.weights)
    ema.apply_model_ema(model)
model.eval(); print(f'model ready in {time.time() - t0:.0f}s', flush=True)
inf = DefaultInferencer(cfg, model)
files = sorted(f for f in os.listdir(a.images) if f.lower().endswith(('.jpg', '.jpeg')))
if a.limit: files = files[:a.limit]
res, t1 = {}, time.time()
for i, f in enumerate(files):
    im = read_image(os.path.join(a.images, f), format='BGR')
    ins = inf(im)['instances'].to('cpu')
    res[f] = dict(height=im.shape[0], width=im.shape[1], bbox=ins.pred_boxes.tensor.tolist(), score=ins.scores.tolist(),
                  class_id=ins.pred_classes.tolist(), **{'class': [classes[c] for c in ins.pred_classes.tolist()]})
    if i % 10 == 0 or i == len(files) - 1:
        print(f'{i + 1}/{len(files)} {f} n={len(ins)} {(time.time() - t1) / (i + 1):.2f}s/img', flush=True)
meta = dict(classes=classes, score_thresh=a.score_thresh, config=a.config, weights=os.path.basename(a.weights),
            torch=torch.__version__, cuda=torch.version.cuda, device=a.device,
            gpu=torch.cuda.get_device_name(0) if a.device == 'cuda' else None,
            n_images=len(files), sec_per_image=(time.time() - t1) / max(1, len(files)), random_init=a.random_init)
json.dump(dict(meta=meta, detections=res), open(a.out, 'w'))
print('wrote', a.out, meta, flush=True)
