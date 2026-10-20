# Auto-labeling feasibility: P(hand is anatomically OK) without manual labeling

Time-boxed study (~1.5 h, CPU only: 8 vCPU, no GPU, no paid APIs). All scripts, raw outputs and logs are in `auto/`.

## TL;DR

* **No single free signal is good enough on its own**, but a cheap ensemble is: **Gemma-4-E4B (GGUF Q4, llama.cpp) asked a plain Yes/No question + ADetailer YOLO hand confidence + CLIP zero-shot** reaches **AUC 0.93** (0.89 on the subset I did not mine with CLIP) on a 35-hand test set (+5 non-hand crops) that I labeled by eye.
* Best single teacher: **Gemma-4-E4B Yes/No, AUC 0.88, 4.8 s/crop → ~7.5 h for 5,604 crops** (one overnight run).
* Qwen2.5-VL-3B/7B were weaker (AUC 0.76 / 0.84) and slower (8.7 / 13 s per crop).
* Ready-made "malformed hand" models from papers (HandCraft YOLO, CountHallu quality classifier) **do not transfer** to our 2025-26 Civitai images (AUC 0.65 / 0.31).
* Simple 2D synthetic negatives (pasted extra finger, inpainted missing finger, mesh warp) look like **edit traces, not diffusion artifacts**. Not recommended as the main source of negatives.
* About **20% of the pre-annotated "hands" are not hands at all** (paws, coral, armour, flowers, faces). A "no hand" gate is needed before any OK/BAD classifier.
* The single best data upgrade is **HAD (Human Artifact Dataset)**: 37.5k AI images with human-drawn artifact boxes (hand is one class), 8.8 GB, downloadable without signup, **but the repo has no license file**.
* For an honest metric the user still has to check **~150–200 crops** with a binary OK/BAD/no-hand click (~15–20 min). These labels are also needed to calibrate the probability. Everything else can be automatic.

**Caveat on the numbers:** n = 35 hands (13 bad / 22 ok). The 95% CI on AUC is roughly ±0.1. Treat the ranking of methods as indicative, not final.

## 1. Test set (40 crops)

* Crops: every pre-annotated box with short side ≥ 56 px (3,263 of 5,604 boxes), square with 30% padding (`auto/make_crops.py`).
* Gold (my own careful visual judgment, `auto/gold40.json`):
  * **13 BAD**: fused or extra fingers, hands melting into objects or clay, blob or mush hands.
  * **22 OK**.
  * **5 NO-HAND**.
* Bad hands are rare: in ~150 random crops I saw roughly 5–8% clearly bad hands and ~20% non-hands. To get 13 bad ones I looked at random sheets and also at sheets sorted by low CLIP score and by MediaPipe failure or odd geometry. 5 of the 13 bad crops came from the CLIP-sorted sheets, so CLIP is also reported on the subset without them ("excl. CLIP-mined").
* Grid with every method's score: **`auto/test40_grid.jpg`** (SFW, checked visually).

## 2. Methods and agreement with my labels (35 hands)

Higher score means "more likely OK". AUC is OK-vs-BAD; bestBalAcc is optimistic because the threshold is tuned on the same 35 crops.

| method | AUC | AUC excl. CLIP-mined | best bal. acc | speed (8 vCPU) |
|---|---:|---:|---:|---:|
| **Gemma-4-E4B-it Q4_0, Yes/No logprob** | **0.88** | **0.84** | 0.84 | **4.8 s/crop** |
| Qwen2.5-VL-7B Q4_K_M, Yes/No logprob | 0.84 | 0.79 | 0.81 | 13 s/crop |
| Gemma-4-E4B, 1–5 rating (leading prompt) | 0.82 | 0.76 | 0.77 | 5.3 s |
| Qwen2.5-VL-7B, 1–5 rating | 0.79 | 0.78 | 0.76 | 12.7 s |
| Qwen2.5-VL-3B Q4_K_M, 1–5 rating | 0.76 | 0.73 | 0.72 | 8.7 s |
| CLIP ViT-B/32 zero-shot (3 good vs 3 bad prompts) | 0.86 | 0.77 | 0.78 | ~15 ms |
| ADetailer `hand_yolov8s`/`v9c` box confidence | 0.81 | 0.81 | 0.79 | free (already computed) |
| MediaPipe Hands: detected or not | 0.65 | 0.58 | 0.71 | ~17 ms |
| MediaPipe bone-length plausibility (z-score) | 0.60 | n/a | 0.62 | — |
| HandCraft YOLO (2-class standard/non-standard hand) | 0.65 | — | 0.66 | ~0.1 s |
| CountHallu RealHand quality classifier (MaxViT) | 0.31 | — | 0.52 | ~0.1 s |
| **rank-avg(Gemma Y/N, ADetailer conf, CLIP)** | **0.93** | **0.89** | — | — |
| rank-avg(Gemma Y/N, ADetailer conf) | 0.89 | 0.88 | — | — |
| rank-avg(Gemma Y/N, Qwen-7B Y/N, ADetailer) | 0.88 | 0.86 | — | — |

Observations:
* **Prompting matters more than model size.** A neutral "Answer Yes/No/None" with logprob P(Yes)/(P(Yes)+P(No)) beats a leading "AI images often have malformed hands… rate 1–5" prompt. With the leading prompt every model rates almost everything 1–2. The VLMs are miscalibrated in both cases. At P(Yes) > 0.5, Gemma catches only 38% of bad hands with 5% false alarms. Thresholds must be set on gold data.
* **Non-hand detection:** VLMs with a "None" option and CLIP "no hand" prompts flag most non-hand crops (Qwen-7B: P(None) ≥ 0.9 on all 5). Qwen-7B also says "None" too often on real hands (mean 0.43), so it is a gate at best. Gemma is the reverse (it misses 1–2 non-hands).
* **What the signals actually detect:** blurry, mushy and blob hands are easy (low detector confidence, low CLIP score). Clean renders with one extra or fused finger (#7, #8, #12 in the grid) are hard for every method. The VLMs catch some of them, CLIP and the detectors mostly do not.
* MediaPipe finds landmarks on only 40% of crops (1,300 / 3,263). It fails on anime, art and gloves, so it is too style-biased to use as a label.
* The bundled `llama-cpp-python` `Qwen25VLChatHandler` emits a stray ":"/"." first token, which breaks logprob scoring. Use the generic `MTMDChatHandler` (`auto/vlm_run.py`).

**Silver-label → student proxy (no human labels used):** take CLIP-B/32 embeddings of all 3,263 crops, label the bottom 15% / top 50% of the teacher score as BAD / OK, and fit a logistic regression. On the gold set this gives **AUC 0.86** with the CLIP teacher and **0.90** with the CLIP+ADetailer teacher. A cheap linear probe keeps the teacher's ranking quality, so the slow VLM teacher only has to run once.

## 3. Synthetic negatives (prototype, `auto/synth.py`, `auto/synthetic_negatives_demo_v2.jpg`)

Using MediaPipe 2D landmarks on clean photo hands I tried three edits:
* (a) duplicate a finger strip and rotate it 24–36° around its MCP joint (extra finger);
* (b) cv2-inpaint a finger (missing finger);
* (c) a Gaussian-windowed random mesh warp around the palm (melted hand).

Verdict:
* Edit (a) sometimes gives a plausible ghost or extra finger, but with visible blending seams.
* Edit (b) leaves smears.
* Edit (c) is either invisible or an obvious wobble.

A classifier trained on these would learn "detect editing traces", not "detect anatomy". It also only works where MediaPipe works (40% of crops, mostly photos). Realistic synthetic negatives would need diffusion inpainting of the hand region with a weak model (e.g. SD1.5 with no hand fixes), which needs a GPU (Kaggle). **Not worth it at this stage.**

## 4. Public datasets (verified)

| dataset | URL | size | labels | license | download w/o signup |
|---|---|---|---|---|---|
| **HAD**: Human Artifact Dataset (Wang et al., 2024, arXiv 2411.13842) | github.com/wangkaihong/HADM → Dropbox zip | **8.8 GB** zip (Content-Length verified); 37,554 images (SDXL, DALL-E 2/3, Midjourney), 84,852 boxes | bounding boxes for local artifacts on 6 body parts **incl. hand**, plus 12 global classes (missing/extra parts) | **no license file in repo** (GitHub API: license = none) → research use, ask authors before redistribution | **yes** (direct Dropbox link). The HADM-L detector weights are also public (5.7 GB EVA-02-L, too heavy for this CPU, not tested) |
| HandCraft (WACV 2025) | kfzyqin.github.io/handcraft, HF Space `zhenyueqin/handcraft` | Space has `model/yolo.pt` (137 MB, classes *standard hand* / *non-standard hand*) | detector weights only. The MalHand dataset mentioned in the paper: **no download link found** | Space tagged `cc` | weights yes. **Tested: AUC 0.65, recall of bad hands 0 at conf 0.5** |
| CountHallu RealHand (arXiv 2510.13080) | huggingface.co/datasets/ShyFoo/CountHallu-dataset-RealHand | 1K–10K real hand photos + masks (1 parquet) | **no defect labels** (positives only) | `other` (card says TODO) | yes, ungated. Their quality classifier (Apache-2.0) tested: AUC 0.31, i.e. no transfer |
| RichHF-18K mirror | huggingface.co/datasets/Exploration/richhf_18k_with_images | 10K–100K rows (Pick-a-Pic images) | per-pixel *implausibility* heatmaps + scores (all artifacts, not hand-specific) | not stated on the mirror | yes, ungated |

The search also turned up the "Detecting and Restoring Non-Standard Hands" paper (arXiv 2312.04236, HaGRID-based). I did not verify that its data is downloadable.

## 5. Recommended fully-automatic pipeline

1. **Gate "is this a hand"** on all 5,604 boxes, using P(None) from the VLM plus CLIP "no hand"/"paw" prompts and the ADetailer confidence. This drops ~20% junk boxes. It doubles as a free improvement of the pre-annotations.
2. **Teacher scores**: Gemma-4-E4B Yes/No logprob on every crop (≈ 7.5 h overnight at 448 px; ≈ 4–5 h at 336 px), rank-averaged with the ADetailer confidence and CLIP zero-shot. Expected AUC ≈ 0.85–0.9 against human labels.
3. **Silver labels**: confident tails only, e.g. bottom ~15% = BAD, top ~50% = OK, middle discarded. Optionally add HAD hand-artifact crops as extra BAD examples, and HAD hands with no overlapping artifact box as OK examples. These are real human labels, but from older generators.
4. **Student**: frozen embeddings (CLIP ViT-L/14 or DINOv2-B, ~10–20 min on CPU for all crops) + logistic regression, or a small MLP. The proxy experiment above keeps AUC ≈ 0.86–0.90. Training a YOLO or a two-stage detector + classifier on Kaggle comes after this, using the same silver labels.
5. **Calibration + honest metric**: the user checks **150–200 crops**, stratified over the score range so bad hands are not just 5% of the set. That is ~15–20 min of OK/BAD/no-hand clicks. Fit Platt/isotonic calibration on half and report AUC, recall on BAD and ECE on the other half. Without this the "probability" is just a ranking, and the README must not claim accuracy.

Compute on this box: ~8 h unattended (VLM) + <30 min for everything else. RAM peak ~6 GB for Gemma-E4B Q4 (fits in 4–6 GB free if nothing else runs; Qwen-7B needs ~7 GB).

Main risk: subtle anatomy errors on otherwise clean renders (one extra or fused finger) stay the weakest category. Expect recall on them to be noticeably lower than on mushy or blurry hands. HAD data or a few dozen user-confirmed examples are the best fix.

## Files

* `auto/crops/` (3,263 crops), `auto/crops.json`, `auto/gold40.json`, `auto/clip_mined_bad.json`
* `auto/vlm_run.py` + `auto/vlm_{qwen3b,qwen7b,gemma4e4b,gemma4e4b_yn,qwen7b_yn}.json` (per-crop scores, seconds)
* `auto/clip_zs.py` → `auto/clip_zs.json`, `auto/clipb32_feats.npy`; `auto/mp_feats.py` → `auto/mp.json`; `auto/weak_eval.py` → `auto/weak40.json`
* `auto/analyze.py`, `auto/combo.py` (all numbers above), `auto/grid40.py` → `auto/test40_grid.jpg`
* `auto/synth.py` → `auto/synthetic_negatives_demo_v2.jpg`
* external weights: `weights/ext/handcraft_yolo.pt`, `weights/ext/counthallu_quality.pth`; VLM GGUFs in `/workspace/vlm/` (~13 GB)
