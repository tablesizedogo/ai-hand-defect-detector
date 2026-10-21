"""HADM-L (Human Artifact Detection Model, local) inference on current PyTorch.

HADM: K. Wang, L. Zhang, J. Zhang, "Detecting Human Artifacts from Text-to-Image Models", arXiv:2411.13842,
https://github.com/wangkaihong/HADM. The HADM repository has no LICENSE file, so this package does NOT vendor or
redistribute its code or weights: the code is cloned from GitHub at a pinned commit and the weights are downloaded from the
authors' own link (only when you allow it). Nothing in the clone is edited except one symlink; everything else needed to
run on torch 2.x is done at runtime by `_install_shims()`:

* xformers' memory_efficient_attention -> torch.nn.functional.scaled_dot_product_attention (same math and scale)
* detectron2's compiled C++ ops are not built: ViTDet / Cascade R-CNN inference only needs torchvision ROIAlign / NMS
* Pillow >= 10 aliases (Image.LINEAR / ANTIALIAS / CUBIC); missing training-only imports (wandb, mmcv.ops.soft_nms,
  fairscale checkpoint_wrapper, and pkg_resources if setuptools no longer ships it) are stubbed; activation checkpointing is off
* `detectron2/model_zoo/configs` -> symlink to the repo's `configs/` (normally created by `pip install -e .`)
* the checkpoint is a pickled training checkpoint: it is loaded with weights_only=False *only inside* this loader
  (the environment variable is restored afterwards) and its size is checked first.
Preprocessing equals the repo's DefaultInferencer (BGR, ResizeShortestEdge(1024, max_size=1024)); the image is JPEG
re-encoded first (q=95) because the authors state the models expect JPEG input. EMA weights are applied as in the repo's
`--inference` mode. test_score_thresh is lowered to 0.001 so weak artifact scores are kept.
"""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

import numpy as np

HADM_REPO = "https://github.com/wangkaihong/HADM.git"
HADM_COMMIT = "d09f66758e169a9271efadf6233c7bb912eb1a9e"
WEIGHTS_NAME = "HADM-L_0249999.pth"
WEIGHTS_URL = ("https://www.dropbox.com/scl/fi/zwasvod906x1akzinnj3i/HADM-L_0249999.pth"
               "?rlkey=bqz5517tm8yt8l6ngzne4xejx&st=k1a1gzph&dl=1")
WEIGHTS_SIZE = 5733865413
CONFIG = "projects/ViTDet/configs/eva2_o365_to_coco/demo_local.py"
CLASSES = ["face", "torso", "arm", "leg", "hand", "feet"]


def ensure_repo(root: Path, allow_clone: bool = True) -> Path:
    repo = Path(root) / "HADM"
    if (repo / CONFIG).exists():
        return repo
    if not allow_clone:
        raise FileNotFoundError(f"HADM code not found in {repo}. Clone it: git clone {HADM_REPO} {repo} && "
                                f"git -C {repo} checkout {HADM_COMMIT}")
    Path(root).mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", HADM_REPO, str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", HADM_COMMIT], check=True)
    return repo


def ensure_weights(root: Path, allow_download: bool = False) -> Path:
    w = Path(root) / WEIGHTS_NAME
    if w.exists() and w.stat().st_size == WEIGHTS_SIZE:
        return w
    if not allow_download:
        raise FileNotFoundError(
            f"HADM-L weights not found at {w}. Download them from the authors' link in the HADM README "
            f"({WEIGHTS_URL}) - 5.7 GB, no license file, research use - and place the file there, "
            f"or enable allow_download in the loader node.")
    Path(root).mkdir(parents=True, exist_ok=True)
    part = w.with_suffix(".part")
    print(f"[hand_fixer] downloading HADM-L weights (5.7 GB) to {w} ...", flush=True)
    with urllib.request.urlopen(urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "Mozilla/5.0"})) as r, open(part, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    if part.stat().st_size != WEIGHTS_SIZE:
        raise RuntimeError(f"unexpected HADM-L weights size {part.stat().st_size} (expected {WEIGHTS_SIZE})")
    part.rename(w)
    return w


def _install_shims(repo: Path):
    if "detectron2" in sys.modules and not str(getattr(sys.modules["detectron2"], "__file__", "")).startswith(str(repo)):
        raise RuntimeError("another detectron2 is already imported in this process; HADM needs its own fork "
                           f"({repo}). Restart without the other package or run HADM in a separate process.")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    mz = repo / "detectron2/model_zoo/configs"
    if not mz.exists():
        try:
            os.symlink(repo / "configs", mz)
        except OSError:  # e.g. Windows without symlink rights
            import shutil
            shutil.copytree(repo / "configs", mz)
    from PIL import Image
    for old, new in (("LINEAR", "BILINEAR"), ("ANTIALIAS", "LANCZOS"), ("CUBIC", "BICUBIC")):
        if not hasattr(Image, old):
            setattr(Image, old, getattr(Image, new))

    def stub(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        sys.modules[name] = mod
        return mod

    try:
        import pkg_resources  # noqa: F401  (removed from recent setuptools; detectron2.model_zoo needs one function)
    except ImportError:
        def _resource_filename(pkg, rel):
            import importlib.util
            return os.path.join(os.path.dirname(importlib.util.find_spec(pkg).origin), rel)
        stub("pkg_resources", resource_filename=_resource_filename)
    try:
        import wandb  # noqa: F401
    except ImportError:
        stub("wandb")
    try:
        import mmcv.ops  # noqa: F401
    except ImportError:
        def _no_soft_nms(*a, **k):
            raise RuntimeError("soft-NMS needs mmcv (not used by the HADM-L config)")
        m = stub("mmcv")
        m.ops = stub("mmcv.ops", soft_nms=_no_soft_nms)
    try:
        import fairscale.nn.checkpoint  # noqa: F401
    except ImportError:
        f = stub("fairscale")
        f.nn = stub("fairscale.nn")
        f.nn.checkpoint = stub("fairscale.nn.checkpoint", checkpoint_wrapper=lambda m, *a, **k: m)


@contextlib.contextmanager
def _pickle_load_allowed():
    old = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = old


class HADMDetector:
    """Callable: RGB uint8 HxWx3 -> list of {'box': [x0,y0,x1,y1], 'score': float, 'cls': str} in input pixels."""

    def __init__(self, root, device="cuda", score_thresh=0.001, allow_download=False, jpeg_quality=95,
                 random_init=False):
        import torch
        import torch.nn.functional as F

        self.root = Path(root)
        repo = ensure_repo(self.root, allow_clone=True)
        weights = None if random_init else ensure_weights(self.root, allow_download)
        _install_shims(repo)
        import detectron2.modeling.backbone.vit as vit
        vit.xops = types.SimpleNamespace(memory_efficient_attention=lambda q, k, v: F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2))
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import LazyConfig, instantiate
        from detectron2.engine.defaults import DefaultInferencer
        from detectron2.modeling import ema

        cfg = LazyConfig.load(str(repo / CONFIG))
        cfg.model.backbone.net.use_act_checkpoint = False
        for p in cfg.model.roi_heads.box_predictors:
            p.test_score_thresh = score_thresh
        cfg.train.device = device
        cfg.train.model_ema.enabled = True
        cfg.train.model_ema.device = device
        cfg.train.model_ema.use_ema_weights_for_eval_only = True
        model = instantiate(cfg.model)
        model.to(device)
        ema.may_build_model_ema(cfg, model)
        if weights is not None:
            with _pickle_load_allowed():
                DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(str(weights))
            ema.apply_model_ema(model)
        model.eval()
        self.inferencer = DefaultInferencer(cfg, model)
        self.jpeg_quality = jpeg_quality
        self.device = device

    def __call__(self, rgb: np.ndarray):
        from PIL import Image
        im = Image.fromarray(rgb)
        if self.jpeg_quality:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=self.jpeg_quality)
            im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        bgr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
        ins = self.inferencer(bgr)["instances"].to("cpu")
        return [dict(box=[float(v) for v in b], score=float(s), cls=CLASSES[int(c)])
                for b, s, c in zip(ins.pred_boxes.tensor.tolist(), ins.scores.tolist(), ins.pred_classes.tolist())]
