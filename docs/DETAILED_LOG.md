# Detailed project log (stages 1–1b, all gold rounds)

> This is the full, chronological write-up that used to be the README. The short version is in [../README.md](../README.md).

> **Status: stage 1 (data) and stage 1b (automatic labeling) are done.** On the independent gold v2 round (100 crops, 8 broken hands, wide CIs), HADM-L alone reaches AUC 0.76 [0.51–0.96]. The v3 fused teacher gets 0.65, and the CLIP-L student 0.59. The 0.79 of the v3 teacher on gold v1 was optimistic: it was chosen on that set. Automatic QC is not solved yet. Ranking by HADM-L is the best triage signal so far. See [Gold-check results](#gold-check-results-honest-metrics).

Diffusion models still break hands: extra or fused fingers, broken wrists, mangled blobs. This project builds a detector that
finds hands in generated images and classifies them as **`hand_ok`** or **`hand_bad`**. It ships as a **ComfyUI custom node** that
masks broken hands and sends them to inpainting automatically, which works as a quality-control step for generation pipelines.

It continues [civitai-multimodal-rag](https://github.com/tablesizedogo/civitai-multimodal-rag): the images come from the same
100k-image Civitai corpus (strict SFW), and candidate selection uses its CLIP index.

## Plan

| stage | what | status |
|---|---|---|
| 0. Problem setup | classes `hand_ok` / `hand_bad`, labeling guide, target metrics: mAP50 and **recall on `hand_bad`** (a missed defect costs more than a false alarm in QC) | done |
| 1. Data | CLIP-guided candidate mining → hand-detector screening → balanced selection of 2,500 images → pre-annotation | done |
| 1b. Automatic labeling | VLM teacher + detector + CLIP → silver labels → linear probe on CLIP-L / DINOv2 (ONNX) → 150-crop user gold check | done; gold metrics below, teacher needs improvement |
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

- Guide: [LABELING_GUIDE.md](../LABELING_GUIDE.md) (classes, defect rules, edge cases, tiny-hand threshold of 24 px).
- Tooling: Label Studio (local files) or CVAT (COCO import). See [docs/LABELING_QUICKSTART.md](LABELING_QUICKSTART.md).
- Pre-boxes in Label Studio (verified on a local Label Studio 1.23 instance: 2,500 tasks and 2,500 predictions imported, images served from local files):

  ![Label Studio](labelstudio_screenshot.jpg)
- Pre-annotation overview (cyan = pre-box, number = detector confidence): [qa/qa_overview.jpg](../qa/qa_overview.jpg)
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

## Gold-check results (honest metrics)

### Headline: gold v2 (independent round, the honest test)

Gold v2 contains 100 new crops judged by the user. It was sampled with the v3 teacher before that round was labeled, and it is never used for training or for any modeling choice.
It is enriched for likely-broken hands: 60 crops come from the lowest 10% of the v3 teacher, 32 are stratified over the rest, and 8 are gate-rejected crops.
Every crop carries a sampling weight (stratum size / crops picked from it), so the metrics describe the full crop population.
An enriched crop has weight 7.0, a crop from the other strata about 117. The 60 enriched crops are heavily over-sampled, so they count less each.

The answers were 67 OK, **8 BAD**, 11 not-a-hand and 14 can't-judge, which leaves **75 gold hands**. The 95% CIs come from 2,000 label-stratified bootstrap resamples.
With 8 positives, one BAD crop moves recall by about 12 points. Full tables: `autolabel_data/gold_check_v2/eval.md` and `analysis.md`.

| score (BAD = positive class), gold v2 | AUC | AP (BAD) | recall BAD @ 5% FPR | recall BAD @ 10% FPR |
|---|---|---|---|---|
| HADM-L hand artifact alone | **0.76 [0.51–0.96]** | 0.34 [0.15–0.76] | 0.25 [0.00–0.80] | 0.28 [0.00–1.00] |
| v3 teacher (HADM + Gemma + det conf) | 0.65 [0.43–0.85] | 0.20 [0.14–0.39] | 0.04 [0.00–0.20] | 0.06 [0.00–0.30] |
| CLIP-L/14 student, v3 silver labels | 0.59 [0.33–0.85] | 0.22 [0.07–0.58] | 0.24 [0.00–0.60] | 0.24 [0.00–0.70] |
| detector confidence | 0.54 [0.33–0.77] | 0.15 [0.09–0.32] | 0.00 [0.00–0.10] | 0.04 [0.00–0.20] |
| CLIP-L/14 student, v2 silver labels | 0.52 [0.28–0.75] | 0.14 [0.08–0.32] | 0.01 [0.00–0.10] | 0.01 [0.00–0.10] |
| Gemma-4-E4B alone | 0.52 [0.23–0.82] | 0.16 [0.08–0.40] | 0.01 [0.00–0.10] | 0.01 [0.00–0.50] |
| DINOv2-base student, v3 / v2 labels | 0.51 / 0.45 | 0.14 / 0.13 | 0.00 / 0.01 | 0.00 / 0.01 |
| v2 teacher (Gemma + det + CLIP), on the 52 hands it covers (7 BAD)\* | 0.56 [0.28–0.85] | 0.19 [0.13–0.45] | 0.06 | 0.07 |
| *random score* | 0.50 | 0.12 | 0.05 | 0.10 |

\* The v2 teacher was built on a 4,325-crop VLM snapshot. On the same 52 hands, the v3 teacher gets 0.65 and HADM 0.77.

- **The 0.79 from gold v1 did not hold.** On v2 the v3 teacher reaches 0.65. Only HADM-L alone stays above 0.7: it got 0.66 on v1 and 0.76 on v2, and pooled over both rounds 0.72 [0.54–0.87].
  Averaging Gemma and detector confidence into the teacher *hurt* on v2. Both are near chance on this round.
- **The enrichment barely enriched.** Of the 43 judged hands from the bottom-10% stratum, only 4 are BAD (9%), against an estimated 12% in the population.
  The teacher's lowest tail consists mostly of stylized, skeletal, robotic and gloved hands that are fine.
- **Pooled v1 + v2** (182 hands, 17 BAD; each round's weights are rescaled to its own sample size; v1 is optimistic for v3):
  HADM 0.72 [0.54–0.87], v3 teacher 0.71 [0.58–0.84], CLIP-L v3 student 0.61 [0.44–0.77], Gemma 0.60, detector conf 0.60.
- **Calibration** (`models/calibration.json`, refit on v1 + v2): Platt on the CLIP-L v3 student, `a=0.268`, `b=2.165`.
  Evaluated leave-one-out, ECE drops from 0.34 to 0.04 and Brier is 0.092. Platt only rescales the score, so the ONNX model is unchanged.
- **Hand gate:** 7 of the 8 gate-rejected crops really are not hands. The eighth (#70) is an OK hand with p_hand 0.79.
  4 of the 11 not-a-hand crops passed the gate with p_hand ≥ 0.98 (#8, #79, #98, #100). Among gate-passed crops, the estimated share of non-hands is 0.7%.

### Gold v1 (first round; **optimistic for the v3 teacher**, which was chosen on it)

**Gold set.** 150 crops, stratified by teacher-score decile and style, judged by the user (one pass, one annotator):
98 OK hands, **9 BAD hands**, 34 "not a hand", 9 "can't judge". OK-vs-BAD metrics use the **107 gold hands** and are re-weighted
by the sampling weights. 95% CIs come from 2,000 label-stratified bootstrap resamples. **With 9 positives the CIs are very wide:
one BAD crop moves recall by ~11 points, and no difference between the scores below is statistically significant.**
The models were trained on silver labels from 4,325 of the 5,604 crops (the VLM job was still running); gold crops are never used for training.

| score (P(ok); BAD = positive class) | AUC | AP (BAD) | recall BAD @ 5% FPR | recall BAD @ 10% FPR |
|---|---|---|---|---|
| Gemma-4-E4B alone | 0.69 [0.49–0.87] | 0.22 [0.11–0.53] | 0.23 [0.00–0.60] | 0.34 [0.00–0.70] |
| fused teacher (Gemma + detector conf + CLIP zero-shot, rank average) | 0.67 [0.46–0.86] | 0.37 [0.11–0.68] | 0.34 [0.00–0.70] | 0.34 [0.10–0.70] |
| **student: CLIP-L/14 linear probe (ONNX)** | **0.66 [0.44–0.85]** | 0.21 [0.10–0.50] | 0.23 [0.00–0.60] | 0.34 [0.00–0.70] |
| student: DINOv2-base linear probe (ONNX) | 0.51 [0.31–0.71] | 0.11 [0.07–0.27] | 0.11 [0.00–0.30] | 0.11 [0.00–0.30] |
| *reference: detector confidence alone* | 0.71 [0.53–0.87] | 0.25 [0.12–0.58] | 0.23 [0.00–0.60] | 0.23 [0.00–0.70] |
| *reference: random score* | 0.50 | 0.08 | 0.05 | 0.10 |

For comparison, the CLIP-L student agrees with its own silver labels at CV AUC 0.96. That number measures how well it copies
the teacher, not how well it finds broken hands. On real labels the teacher itself reaches only ≈ 0.67–0.69.

**Calibration.** The raw student probabilities are badly over-confident about BAD (ECE 0.32). Platt scaling on the gold hands,
evaluated leave-one-out, brings ECE to 0.01 [0.01–0.07] and Brier from 0.23 to 0.078. That is no better than always predicting the
base rate (Brier 0.077): after honest calibration the model's calibrated P(ok) spans only ≈ 0.74–0.98 (1st–99th percentile over all gated hands). Isotonic regression was worse
(ECE 0.05) with so few positives. The deployed calibrator is Platt, in `models/calibration.json`.

**"Not a hand" gate.** 34 of 150 gold crops were not hands (feet, armor, hair, animals, objects), versus 25 that the gate had
sampled as non-hands. Gemma's P(hand) separates them well (AUC 0.98), but the original threshold of 0.5 was too lax.

| gate | precision | recall | non-hands left among hands | OK hands dropped |
|---|---|---|---|---|
| P(hand) < 0.5 (before) | 0.92 | 0.65 | 12 (9.9% of passed crops) | 1 of 98 |
| **P(hand) < 0.95 (now)** | 0.89 | 0.91 | 3 (2.8%) | 3 of 98 |

The threshold was chosen on the same gold set, so the "now" row is somewhat optimistic. The gap is large, though:
9 of the 12 missed non-hands had P(hand) 0.71–0.92, while 95 of the 97 OK hands that passed the old gate had P(hand) ≥ 0.95. The stricter gate removed 159 crops,
including about a fifth of the old silver-BAD tail. AUC on hands stayed the same (0.661 → 0.664), so the gate cleans the data but
does not fix the teacher. Gating on detector confidence did not help.

**Failure modes (from the 9 BAD and the top false alarms):**
- **Small crops.** Source images are 512 px, and the median BAD crop is 90 px versus 112 px for OK. Crop size alone gives AUC 0.65,
  so part of the signal is "small and blurry", not anatomy.
- **Subtle defects on grips.** Wrong finger counts or fused fingers on a fist or grip (#20, #38, #118, #147) are ranked as clean
  by every score. #147, a photo-style hand holding a bag strap, is in the top 10% "most OK" for Gemma and both students.
- **Gloves and armor** (#8, #10, #30) get mixed results.
- **False alarms on stylized hands.** Cartoon four-finger hands, creature, robot and skeletal hands, and heavily occluded hands are
  flagged as broken. The teacher confuses "unusual" with "malformed".
- **DINOv2 features carry almost no defect signal here.** CLIP-L is the only usable student backbone.

### External baselines

Two ready-made hand-quality models from CountHallu ([arXiv:2510.13080](https://arxiv.org/abs/2510.13080)), trained on the RealHand
dataset of SD-generated close-up hands, were scored on the same 107 gold hands with the same weights and bootstrap
(`autolabel/08_eval_external.py`, full table in `autolabel_data/gold_check/eval_external.md`). Each model saw the original gold crop
(1.3× box) and a 2.0× context crop from the same 512-px image; the better variant is shown.

| score (BAD = positive class) | AUC | AP (BAD) | recall BAD @ 5% FPR | recall BAD @ 10% FPR |
|---|---|---|---|---|
| [quality classifier](https://huggingface.co/ShyFoo/CountHallu-quality_cls_model-RealHand) P(clean), MaxViT-S, Apache-2.0 (2.0× crop) | 0.57 [0.36–0.77] | 0.17 [0.08–0.46] | 0.13 [0.00–0.40] | 0.23 [0.00–0.50] |
| [finger counter](https://huggingface.co/ShyFoo/CountHallu-counting_model-RealHand) −\|n − 5\|, YOLO, AGPL-3.0 (2.0× crop) | 0.64 [0.46–0.81] | 0.14 [0.08–0.35] | 0.11 [0.00–0.30] | 0.11 [0.00–0.30] |
| finger counter, soft P(n = 5) from box confidences | 0.50 [0.30–0.67] | 0.09 [0.06–0.13] | 0.00 | 0.00 |
| *ours: CLIP-L/14 student (for comparison)* | 0.66 [0.44–0.85] | 0.21 [0.10–0.50] | 0.23 [0.00–0.60] | 0.34 [0.00–0.70] |

Neither transfers to Civitai hands. The quality classifier is near chance and, at its own 0.5 threshold, calls 61–77% of OK hands
"failed". The finger counter finds exactly 5 fingertips on only 1 of 98 OK hands, because most hands in real images are partly
occluded or seen from the side. So the paper's rule "bad if n ≠ 5" flags almost every hand. Its small AUC edge comes from BAD crops more often having 0 detected fingertips (4 of 9 versus 32 of 98 OK
at 2.0×), a "hard to read" signal rather than a real count. Both models expect frontal, open, full-frame hands. That supports the plan: a better teacher needs in-domain data.

### HADM-L (Adobe human-artifact detector)

[HADM-L](https://github.com/wangkaihong/HADM) ([arXiv:2411.13842](https://arxiv.org/abs/2411.13842)) is a ViTDet / EVA-02-L
Cascade R-CNN that detects artifact regions on generated people. One of its classes is `hand`. It ran on a private Kaggle T4
on the **full-resolution Civitai originals** of the gold images (144 of 145; long side 1024), at 1.6 s per image and about
6.5 GPU-minutes in total. It needed a port to current PyTorch, with no xformers and no compiled detectron2 ops; see
`autolabel/09_hadm_kaggle/README.md`. A crop's score is the highest `hand`-artifact score among detections that match the crop
box. Several matching rules give nearly the same result (full table in `autolabel_data/gold_check/eval_hadm.md`).

| score (BAD = positive class) | AUC | AP (BAD) | recall BAD @ 5% FPR | recall BAD @ 10% FPR |
|---|---|---|---|---|
| HADM-L hand artifact, detection inside the crop (≥ 50% of its area) | 0.66 [0.39–0.87] | 0.34 [0.11–0.63] | 0.35 [0.00–0.80] | 0.46 [0.10–0.80] |
| HADM-L hand artifact, IoU ≥ 0.1 with the hand box | 0.65 [0.38–0.87] | 0.32 [0.10–0.61] | 0.13 [0.00–0.70] | 0.46 [0.10–0.80] |
| HADM-L, max over the whole image (no localisation) | 0.62 [0.36–0.84] | 0.22 [0.09–0.53] | 0.13 [0.00–0.60] | 0.35 [0.00–0.80] |
| *ours: detector confidence / Gemma / CLIP-L student* | 0.71 / 0.69 / 0.66 | 0.25 / 0.22 / 0.21 | 0.23 / 0.23 / 0.23 | 0.23 / 0.34 / 0.34 |
| rank average of HADM-L + Gemma + detector conf\* | 0.79 [0.65–0.91] | 0.32 [0.14–0.60] | 0.23 [0.00–0.60] | 0.23 [0.00–0.80] |

\* The fusion was chosen after seeing the gold results, so it is optimistic. For comparison, Gemma + detector conf alone reached 0.72.
In a paired bootstrap it beats the CLIP-L student by +0.12 AUC [−0.06, +0.32]. It beats Gemma alone in 87% of resamples.

- **Alone, HADM-L is not a better teacher.** Its AUC is 0.65–0.66, the same as the student and below detector confidence.
  The paired difference to every one of our scores has a CI spanning roughly ±0.4. It flags many OK hands too: the median OK
  hand has a hand-artifact score of 0.40.
- **It is complementary.** HADM-L ranks the subtle grip and finger-count errors that every other score missed among the top
  8%: #20, #30 and #38. It also puts #147 in the top 21%; every other score called that hand clean.
  It misses the gross, blurry broken hands that Gemma and the detector catch (#1, #8, #10). This is why the fusion helps.
- **By style** (2 / 2 / 5 BAD, anecdotal): anime 0.73, art 0.96, photo 0.65. On anime, HADM-L is the best single score here; Gemma gets 0.68 and the student 0.55.
- Verdict: HADM-L is worth adding as a **third teacher component**, not as a replacement. Scoring all 2,500 source images would take about 70 T4-minutes.
  Proving the gain still needs more BAD gold examples.

### v3 teacher: HADM-L + Gemma + detector confidence (full set)

HADM-L was run on all 2,500 source images behind the 5,604 crops, in about 35 min of Kaggle 2×T4 time. The Gemma job had also
finished all crops. The fusion was then rebuilt on the full set with the new teacher. It averages the ranks of the HADM-L hand-artifact
score (detection ≥ 50% inside the crop), Gemma p_ok and detector confidence; CLIP zero-shot is dropped. The hand gate stays at Gemma p_hand ≥ 0.95. Run with `04_fuse.py --teacher hadm_gemma_det`.
The old outputs are kept in `autolabel_data/v2_gate095_snap4325/` and `models/v2_gate095_snap4325/`.

**Silver labels:** 4,314 of 5,604 crops pass the gate. The training pool has 4,075 hands, after leaving out gold v1, gold v2 and the 40 study crops.
That gives 612 BAD and 2,038 OK, with 1,425 discarded. Silver CV AUC: CLIP-L 0.895 (v2: 0.957), DINOv2 0.874. The new teacher is harder to imitate from frozen embeddings.

Results on gold v1 (107 hands, 9 BAD). **The v3 teacher was chosen after looking at gold v1, so its v1 numbers are optimistic.**

| score (BAD = positive class) | AUC | AP (BAD) | recall BAD @ 5% FPR | recall BAD @ 10% FPR |
|---|---|---|---|---|
| v3 teacher (HADM + Gemma + det conf)\* | 0.79 [0.66–0.91] | 0.35 [0.15–0.64] | 0.23 [0.00–0.70] | 0.34 [0.10–0.80] |
| v2 teacher (Gemma + det conf + CLIP zero-shot) | 0.67 [0.46–0.86] | 0.37 [0.11–0.68] | 0.34 [0.00–0.70] | 0.34 [0.10–0.70] |
| **CLIP-L/14 student, v3 silver labels**\* | 0.67 [0.44–0.85] | 0.28 [0.10–0.57] | 0.35 [0.00–0.60] | 0.35 [0.00–0.70] |
| CLIP-L/14 student, v2 silver labels | 0.66 [0.44–0.85] | 0.21 [0.10–0.50] | 0.23 [0.00–0.60] | 0.34 [0.00–0.70] |
| DINOv2-base student, v3 / v2 | 0.56 / 0.51 | 0.11 / 0.11 | 0.11 / 0.11 | 0.11 / 0.11 |

\* optimistic on gold v1, see above.

- **The student does not inherit the teacher's gain.** Its AUC moves from 0.664 to 0.669. With frozen CLIP-L features and a
  linear head, it can't see what HADM sees: #20, #38 and #147 stay in the "clean" half for the student, but HADM flags them.
  Getting there needs a fine-tuned backbone on higher-resolution crops, or running HADM at inference time.
- **Calibration.** Platt scaling on gold v1, evaluated leave-one-out, brought ECE from 0.37 to 0.01 (`a=0.420`, `b=2.400`). It has since been refit on v1 + v2; see the top of this section.
- **Gold v2 has been labeled.** Its results are at the top of this section; the v3 teacher reaches only 0.65 there. Gold v2 lives in `autolabel_data/gold_check_v2/`: 100 new crops on 4 sheets. 60 come from the lowest 10% of the v3 teacher, 32 are stratified
  over the rest, and 8 are gate-rejected crops. It was sampled before the v3 students were trained, and they never train on it, so it is an honest test of the v3 teacher and students.
  `07_ingest_eval.py --gold-dir gold_check_v2` evaluates v2 alone. `--gold-dir gold_check,gold_check_v2` pools both rounds, rescaling each round's weights to its own sample size.

**Bottom line.** The pipeline is a useful *triage* tool: sort crops, and at a 10% false-alarm rate about a third of broken hands
are caught. It is not yet an automatic QC gate. Next steps: (1) a stronger or ensembled teacher on upscaled crops, for example a
larger VLM, several prompts, or crops from full-resolution images; (2) a second gold round enriched with low-score crops (active
learning), because 9 positives are too few to tune anything; (3) optionally, real hand-artifact examples from HAD (research-only license).

## Data and licensing

The images belong to their Civitai authors and are **not** redistributed in this repository. `meta/images.jsonl` in the labeling
package links every image to its Civitai page. Labels will be published without the images.

## Results

*To be filled after stage 2.*

## Stage 1b: automatic labeling (no manual box labeling)

Manual labeling of 5,604 boxes was replaced by an automatic teacher → student pipeline. A local VLM (Gemma-4-E4B, CPU) gives a Yes/No/None judgment per crop. It is fused with the detector confidence and CLIP zero-shot, and the confident tails become silver labels for a linear probe on CLIP-L / DINOv2 embeddings (exported to ONNX). The user checks only ~150 stratified crops, which are used for calibration and an honest metric.
See [`AUTO_LABEL_FEASIBILITY.md`](../AUTO_LABEL_FEASIBILITY.md) (method comparison on 40 hand-judged crops) and [`AUTO_LABEL_RUN.md`](../AUTO_LABEL_RUN.md) (how to run/resume).
