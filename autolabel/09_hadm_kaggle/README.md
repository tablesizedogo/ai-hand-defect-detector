# Step 9: HADM-L as a teacher candidate (Kaggle GPU)

[HADM](https://github.com/wangkaihong/HADM) (Wang, Zhang, Zhang, arXiv:2411.13842) is a human-artifact detector:
ViTDet / EVA-02-L Cascade R-CNN, trained on the Human Artifact Dataset. The local model HADM-L has 6 artifact classes
(face, torso, arm, leg, **hand**, feet). Weights `HADM-L_0249999.pth` (5.73 GB, Dropbox link in the HADM README); no license file.

| file | what |
|---|---|
| `01_build_input.py` | source images behind the 150 gold crops: Civitai originals (`original=true` CDN variant), JPEG q95, long side ≤ 1024; crop boxes rescaled → `autolabel_data/hadm/input/` (uploaded as a **private** Kaggle dataset) |
| `hadm_infer.py` | HADM-L inference on a JPEG folder → `{image: {bbox, score, class}}` JSON. Also shipped inside the dataset |
| `kernel_gold/` | Kaggle kernel (script, **private**, GPU T4, internet on): pip deps, clone HADM @ `d09f667`, download weights, run `hadm_infer.py` |
| `03_eval_hadm.py` | matching rules + gold evaluation, same protocol as `07_ingest_eval.py` → `gold_check/eval_hadm.{md,json}` |

## Running on a current Kaggle image (no torch 1.12 / xformers / compiled detectron2)

The repo pins Python 3.8, torch 1.12.1+cu116, xformers 0.0.18, mmcv and apex. Kaggle's image (Python 3.13, torch 2.11+cu128)
has no wheels for that stack, so the code was adapted at runtime instead (see the docstring of `hadm_infer.py`):

- The HADM repo is put on `sys.path` and **not compiled**. Inference only needs torchvision ROIAlign/NMS; `detectron2._C` is
  only used for deformable conv, rotated boxes and fast COCO eval. `detectron2/model_zoo/configs` is symlinked to `configs/`.
- `xformers.ops.memory_efficient_attention` → `torch.nn.functional.scaled_dot_product_attention` (same math and scale).
- Pillow ≥ 10 aliases (`Image.LINEAR`, `ANTIALIAS`, `CUBIC`) restored; `wandb` and `mmcv.ops.soft_nms` stubbed (soft-NMS is off in this config).
- `pip install fvcore iopath omegaconf hydra-core pycocotools` + `fairscale --no-deps`. Activation checkpointing off.
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`, because torch ≥ 2.6 defaults `torch.load(weights_only=True)`.
- Loading = the repo's `--inference` path: EMA weights applied. The checkpoint loaded with no missing or unexpected keys.
- `test_score_thresh` lowered from 0.05 to 0.001 in all three cascade stages, so weak artifact scores are kept.
- Preprocessing = the repo's `DefaultInferencer` (BGR, `ResizeShortestEdge(1024, max_size=1024)`), fp32.

Smoke-tested first on the box CPU with random weights (`--random-init --device cpu`, ~21 s/image on 2 threads).

## Run log (2026-10-21)

- Kernel `artem4767/hand-defects-hadm-gold` version 1 (private), dataset `artem4767/hand-defects-hadm-gold-input` (private, 2 versions).
- First push worked. GPU and internet were available. The session took about 6.5 min: pip 20 s, weights 84 s, model load 10 s, inference 265 s.
  Inference ran on 145 images at 1.61 s/image on one T4.
- Input: 144 of 145 images are Civitai originals (832–5120 px, downscaled to 1024 long side). 1 image (`144394553`) has no larger original, so the local 512-px copy is used, upscaled by HADM.
  The rescaled crop boxes match the local crops (median pixel correlation 0.99).

```bash
cd autolabel && ../.venv/bin/python 09_hadm_kaggle/01_build_input.py
# kaggle CLI in its own venv: KAGGLE_API_TOKEN / KAGGLE_USERNAME from the environment, never written to the repo
kaggle datasets create -p ../autolabel_data/hadm/input -r zip          # private by default
kaggle kernels push -p 09_hadm_kaggle/kernel_gold -t 3600
kaggle kernels status artem4767/hand-defects-hadm-gold
kaggle kernels output artem4767/hand-defects-hadm-gold -p ../autolabel_data/hadm/out_gold
../.venv/bin/python 09_hadm_kaggle/03_eval_hadm.py                     # ~3.5 min (2000 bootstraps)
```

Results: `autolabel_data/gold_check/eval_hadm.md`, summary in the main README ("HADM-L").

## Scoring all 5,604 crops (done 2026-10-21, see AUTO_LABEL_RUN.md)

`01_build_input.py --all` → private dataset `artem4767/hand-defects-hadm-all-input`; `kernel_all/` (2 shards on 2×T4, 34.5 min session); `04_crop_scores.py` → `autolabel_data/hadm_hand.jsonl`.

Original estimate:

The 5,604 crops come from 2,500 source images. At 1.61 s/image that is ~67 min of T4 inference plus ~2 min setup, or ~35 min
if the two T4s each take half. A full-set kernel would be `kernel_gold` with a different input dataset. Gold-set verdict: HADM-L
alone is not better than the current signals, so the full run is only worth it as a third fusion component.
