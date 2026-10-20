# Hand-defect detector for AI-generated images

> **Status: stage 1 (data preparation) is complete; manual labeling is in progress.** Metrics below are placeholders until the labels are in.

Diffusion models still break hands: extra or fused fingers, broken wrists, mangled blobs. This project builds a detector that
finds hands in generated images and classifies them as **`hand_ok`** or **`hand_bad`**. It ships as a **ComfyUI custom node** that
masks broken hands and sends them to inpainting automatically, which works as a quality-control step for generation pipelines.

It continues [civitai-multimodal-rag](https://github.com/tablesizedogo/civitai-multimodal-rag): the images come from the same
100k-image Civitai corpus (strict SFW), and candidate selection uses its CLIP index.

## Plan

| stage | what | status |
|---|---|---|
| 0. Problem setup | classes `hand_ok` / `hand_bad`, labeling guide, target metrics: mAP50 and **recall on `hand_bad`** (a missed defect costs more than a false alarm in QC) | done |
| 1. Data | CLIP-guided candidate mining → hand-detector screening → balanced selection of 2,500 images → pre-annotation → manual labeling with a self-agreement check | pre-annotation done, labeling in progress |
| 2. Training (Kaggle T4/P100) | zero-shot baseline (Grounding DINO "deformed hand"); YOLO11 vs RT-DETR; two-stage detector + crop classifier; learning curve at 250/500/1000/all labels; error analysis by base model and style | planned |
| 3. ComfyUI node | detect → dilated mask → inpaint with the same prompt, up to N retries; evaluate the share of defect-free images before/after on 200 fresh generations | planned |
| 4. Packaging | ONNX export (no PyTorch in the node), weights and labels on the HF Hub (no images), README with GIF | planned |

## Stage 1: data pipeline

```
100,167 SFW Civitai images (webp, 512 px)                       ← civitai-multimodal-rag corpus
  │ 60 CLIP text queries ("holding a cup", "praying hands", "anime character waving", …), top-900 each
  ▼ 20,684 candidates
  │ filters: clean_flags (junk/near-dups), EN+RU prompt blocklist, extra prompt terms, 3 CLIP zero-shot SFW filters
  ▼ 13,809 candidates
  │ hand_yolov8s screening: ≥1 hand with conf ≥ 0.35 and short side ≥ 40 px, ≤ 6 hands
  ▼ 8,949-image pool (fewer after the stricter risk filter)
  │ balanced by base-model family × style, ≤ 3 images per author, CLIP near-dup cos ≥ 0.92 removed,
  │ CLIP "fan-service" risk score ≤ 0.022 + 39 images removed by manual visual review
  ▼ 2,500 images → pre-annotation (YOLOv8s ∪ YOLOv9c, NMS) → 5,604 boxes, all `hand_ok`
```

**Selected images** (`data/selection_stats.json`):

| family | photo | anime | art | total |
|---|---:|---:|---:|---:|
| Krea 2 | 433 | 312 | 339 | 1,084 |
| Illustrious | 98 | 433 | 195 | 726 |
| Flux (all versions) | 211 | 46 | 141 | 398 |
| Other (Anima, Z-Image, SDXL, Pony, MOSS) | 78 | 146 | 68 | 292 |
| **total** | **820** | **937** | **743** | **2,500** |

Images come from 1,494 authors. Flux and the smaller families are under their quota (600/500) because of the
3-per-author cap and the SFW filters; the shortfall was filled with Krea 2. The style (photo/anime/art) is a CLIP zero-shot label.

### Pre-annotator benchmark

We ran the detectors on 48 images (16 photo, 16 anime, 16 art) on CPU. The GT is the union of all detectors' boxes,
checked visually; 7 ambiguous images were excluded, leaving 54 hands. Recall is therefore *relative to what at least
one detector found*.

| detector | s/img (CPU) | recall photo | recall anime | recall art | recall all | precision all |
|---|---:|---:|---:|---:|---:|---:|
| ADetailer `hand_yolov8n` @0.25 | 0.026 | 0.78 | 0.75 | 0.67 | 0.74 | 0.88 |
| ADetailer `hand_yolov8s` @0.25 | 0.038 | 0.91 | 0.75 | 0.80 | 0.83 | 0.94 |
| ADetailer `hand_yolov9c` @0.25 | 0.101 | 0.87 | 0.69 | 0.87 | 0.81 | 0.94 |
| MediaPipe Hands (landmarks → box) | 0.022 | 0.78 | 0.38 | 0.40 | 0.56 | 1.00 |
| Grounding DINO tiny, "a hand." @0.3 | 2.757 | 1.00 | 0.88 | 0.87 | 0.93 | 0.70 |
| **v8s @0.15 ∪ v9c @0.25 (NMS 0.4), chosen** | 0.14 | 0.96 | 0.94 | 1.00 | **0.96** | **0.86** |

MediaPipe fails on anime and art. Grounding DINO has the best single-model recall, but it is about 20× slower and
fires on whole people, feet and objects. The YOLO union gives nearly the same recall at about 0.14 s per image.

Weights: [Bingsu/adetailer](https://huggingface.co/Bingsu/adetailer) (`hand_yolov8s.pt`, `hand_yolov9c.pt`).

### Labeling

- Guide: [LABELING_GUIDE.md](LABELING_GUIDE.md) (classes, defect rules, edge cases, tiny-hand threshold of 24 px).
- Tooling: Label Studio (local files) or CVAT (COCO import). See [docs/LABELING_QUICKSTART.md](docs/LABELING_QUICKSTART.md).
- Pre-boxes in Label Studio (verified on a local Label Studio 1.23 instance: 2,500 tasks and 2,500 predictions imported, images served from local files):

  ![Label Studio](docs/labelstudio_screenshot.jpg)
- Pre-annotation overview (cyan = pre-box, number = detector confidence): [qa/qa_overview.jpg](qa/qa_overview.jpg)
- Self-agreement: 120 random images are relabeled from scratch about a week later.
- The train/val/test split will be made **by author** (and checked for prompt near-duplicates), so that series from one author
  do not leak into the test set.

## Reproduce

```bash
# 1a. candidates (runs in the civitai-multimodal-rag search venv: needs its CLIP index)
cd ../civitai-multimodal-rag/search && .venv/bin/python ../../hand_defects/scripts/01_select_candidates.py
# everything else uses this project's venv
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/02_benchmark.py         # detector benchmark (downloads weights into weights/)
.venv/bin/python scripts/03_detect_candidates.py hand_yolov8s.pt data/candidates.jsonl data/det_v8s.jsonl 0.15
.venv/bin/python scripts/04_select_final.py      # balanced selection
.venv/bin/python scripts/05_preannotate.py       # boxes + weak CLIP defect hint
.venv/bin/python scripts/07_qa_grids.py 0        # QA grids
.venv/bin/python scripts/08_guide_examples.py    # guide example sheets
.venv/bin/python scripts/06_package.py           # dist/hand_defects_labeling.zip
```

## Data and licensing

The images belong to their Civitai authors and are **not** redistributed in this repository. `meta/images.jsonl` in the labeling
package links every image to its Civitai page. Labels will be published without the images.

## Results

*To be filled after stage 2.*
