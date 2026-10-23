# Auto-labeling run: status, restart and next steps

The pipeline was chosen in `AUTO_LABEL_FEASIBILITY.md`:

1. Gemma-4-E4B answers Yes/No/None for every crop, which gives a teacher score and an "is it a hand" gate.
2. That score is fused with the detector confidence and CLIP zero-shot.
3. Silver labels are taken from the confident tails of the fused score.
4. A student classifier is trained on frozen embeddings.
5. A ~150-crop gold check by the user gives calibration and an honest metric.

All code is in `autolabel/`. All derived data is in `autolabel_data/` (git-ignored). Models go to `models/`.
Python environments:
- `.venv` (torch, open_clip, sklearn, onnx) for every step **except** step 3;
- `/workspace/rag_civitai/search/.venv` (llama-cpp-python 0.3.36) for step 3.

## Status (written 2026-10-20 ~14:50 MSK)

| step | script | status |
|---|---|---|
| 1. crops + manifest | `autolabel/01_crop_all.py` | **done**: 5,604 crops in `autolabel_data/crops/` (square, 30% padding, max side 448). Metadata in `autolabel_data/manifest.jsonl` |
| 2. cheap scores + embeddings | `autolabel/02_cheap_scores.py` | **done**: `clip_b32_zs.jsonl`, `emb_clip_b32.npy` (N×512), `emb_dinov2_base.npy` (N×768), `emb_clip_l14.npy` (N×768), row order in `keys.json`. The detector confidence is in the manifest (`det_conf`) |
| 3. Gemma-4-E4B Yes/No/None | `autolabel/03_vlm_score.py` via `autolabel/run_vlm.sh` | **running, detached** since 15:15 MSK with 6 threads (8 threads was not faster). Loop PID 346615, worker PID 346618 (PIDs change after a restart; see `vlm_job.pid` / `pgrep`). ~5.2 s/crop → **ETA ≈ 23:15 MSK 2026-10-20** |
| 4. fusion + hand gate | `autolabel/04_fuse.py` | ready, tested (dry run) |
| 5. silver labels + student + ONNX | `autolabel/05_silver_train.py` | ready, tested (dry run, CLIP-B/32: logreg + MLP, ONNX parity 2e-7) |
| 6. gold-check set + contact sheets | `autolabel/06_goldset.py` | ready, tested (dry run) |
| 7. ingest answers → calibration + eval | `autolabel/07_ingest_eval.py` | ready, tested with fake answers |

Why there was a pause at the start: running torch embeddings next to llama.cpp made Gemma go from 5 s to 25–40 s per crop because of CPU oversubscription. **Never run heavy CPU jobs while step 3 runs.** If you must, pause the worker first: `kill -STOP <worker pid>`, and later `kill -CONT <worker pid>`.

## Partial run on a VLM snapshot (2026-10-21 ~17:05 MSK)

The user decided not to wait for the full VLM job. Steps 4–6 were run on the crops that already had a Gemma answer.
The VLM job was **not** stopped: it keeps appending to `vlm_gemma4e4b.jsonl`.

**Snapshot.** At 17:02 MSK the file was copied to `autolabel_data/vlm_gemma4e4b_snapshot_20261021.jsonl`. All lines were complete. Details:
- 4,325 of 5,604 crops, 0 errors;
- sha256 `0d2feae9…edce3ac`, also stored in `fused_meta.json`.

The style mix of the scored crops is close to the full set: anime 1,592 / art 1,466 / photo 1,267, versus 1,997 / 1,928 / 1,679.

**Commands used** (2 BLAS threads + `nice`, so the Gemma worker is not slowed down; peak extra RAM was well under 1 GB):

```bash
cd /workspace/hand_defects/autolabel
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 ../.venv/bin/python 04_fuse.py --vlm ../autolabel_data/vlm_gemma4e4b_snapshot_20261021.jsonl --scored-only
nice -n 10 ../.venv/bin/python 06_goldset.py --allow-partial --scored-only --replace 146110771_1,145814100_1
nice -n 10 ../.venv/bin/python 05_silver_train.py --emb clip_l14    --model logreg --allow-partial
nice -n 10 ../.venv/bin/python 05_silver_train.py --emb dinov2_base --model logreg --allow-partial
```

**Code changes for partial runs:**
- `04_fuse.py`:
  - `--vlm <file>` selects the VLM results file.
  - `--scored-only` rank-normalises all three signals over the VLM-scored crops only. Unscored crops get `teacher_ok=None` and `is_hand=None`.
  - Writes `fused_meta.json` with the snapshot name, its hash and the counts.
- `05_silver_train.py`:
  - Skips crops with no teacher score.
  - Reports more silver-CV metrics (AP, recall at FPR, Brier, ECE) and stores them in `student_<tag>_meta.json`.
  - The study-40 sanity check uses only the scored study crops.
- `06_goldset.py`:
  - `--scored-only` samples hands only from scored crops.
  - `--replace k1,k2` swaps rejected crops (SFW check) in place for a crop of the same stratum and style. Numbering and all other items stay the same. The swaps are logged in `manifest.json`.
  - Bigger tiles (300 px) and 40 px numbers in a boxed label, so each sheet is 1500×1500.

**Counts:**
- VLM-scored crops: 4,325.
- `is_hand`: 3,519. Not a hand: 806.
- Silver pool: 3,367 hands, after excluding the 150 gold crops and the 40 study crops.
- Silver labels: 505 BAD (teacher ≤ 0.348) and 1,684 OK (teacher ≥ 0.555). The 1,178 crops in between were discarded.

**Gold-check set** (`autolabel_data/gold_check/`):
- 150 crops in `sheet_01.jpg` … `sheet_06.jpg`, 25 per sheet, numbered 1–150 globally.
- 125 hands, stratified by teacher decile and style, plus 25 crops the gate called "not a hand".
- The mapping is `manifest.json`, read by `07_ingest_eval.py`. The answer template is `answers_TEMPLATE.txt`.

**SFW check.** Every sheet was inspected visually. Two crops were replaced:
- #63, `146110771_1`: anime hands on the chest area;
- #107, `145814100_1`: hand on a bare thigh.

Their replacements (`145950231_2`, `128680921_1`) were checked too. Everything else is hands with objects, gloves, weapons and instruments, plus a few non-hands (animals, objects, scenery).

`07_ingest_eval.py` was smoke-tested end-to-end on fake answers. Those outputs were deleted.

**Silver metrics.** These are 5-fold CV against the **silver labels**, so they measure agreement with the teacher, **not ground truth**. The honest numbers will come from the user's gold check via `07`. Logreg students:

| backbone | AUC | acc@0.5 | AP (BAD) | recall BAD @5% FPR | @10% FPR | Brier | ECE (raw) | study-40 sanity AUC* |
|---|---|---|---|---|---|---|---|---|
| CLIP-L/14 | 0.965 | 0.911 | 0.921 | 0.834 | 0.915 | 0.080 | 0.134 | 0.872 (teacher 0.944) |
| DINOv2-base | 0.945 | 0.880 | 0.872 | 0.760 | 0.838 | 0.092 | 0.123 | 0.733 (teacher 0.944) |

\* The study-40 sanity AUC uses my own labels on the 28 study hands that have a VLM score. Very small, not honest, for orientation only.

**Notes on these numbers:**
- Raw probabilities are not calibrated. `class_weight='balanced'` shifts them, hence the ECE above.
- Calibration (Platt and isotonic) is fitted on the gold answers in `07`, which writes `models/calibration.json`.
- ONNX files are `models/hand_ok_clip_l14_logreg.onnx` and `models/hand_ok_dinov2_base_logreg.onnx`. ONNX–sklearn parity is ≤ 3e-7.

### Adding the remaining ~1,280 crops later

When the VLM job prints `ALL DONE`, re-run fusion and training on the full file:

```bash
../.venv/bin/python 04_fuse.py            # full vlm_gemma4e4b.jsonl, no snapshot (gate default --hand-thr 0.95)
../.venv/bin/python 05_silver_train.py --emb clip_l14 --model logreg
../.venv/bin/python 05_silver_train.py --emb dinov2_base --model logreg
```

**Do NOT re-run `06_goldset.py` once the sheets have been sent.** It would renumber the set. The existing `gold_check/manifest.json` is kept, and `05` keeps excluding those crops from training.

The gold sampling weights describe the snapshot population (3,519 hands). That is fine because the scored subset is a large, style-balanced sample of the full set. Re-run `07_ingest_eval.py` after retraining to get the gold metrics for the new students.

## Gold check ingested (2026-10-21 ~19:45 MSK)

**VLM job state at that time:** **not finished**, 5,368 of 5,604 lines, ~18 s/crop because the box was busy. So the existing snapshot (4,325 crops) was used. Fusion was **not** re-run on the full file.

**Answers.** `autolabel_data/gold_check/answers.txt` contains the user's answer verbatim:
- 98 OK, 9 BAD, 34 nohand, 9 skip;
- that gives 107 gold hands.

The gold set was **not** rebuilt or renumbered.

**Gate fix: `04_fuse.py --hand-thr`, default now 0.95 (was 0.5).**
- The user marked 34 nohand. The 0.5 gate had let 12 of them through as hands. 9 of those 12 had p_hand between 0.71 and 0.92.
- Rule `p_hand < 0.95` on the gold set:
  - nohand recall goes from 0.65 to 0.91, precision from 0.92 to 0.89;
  - non-hands among passed crops go from 9.9% to 2.8%;
  - OK hands dropped go from 1 to 3 of 98.
- On the 4,325-crop snapshot, `is_hand` drops from 3,519 to 3,360. 111 of the 159 removed crops were in the old silver-BAD tail.
- `teacher_ok` is unchanged, because ranks are computed over all scored crops.
- The students were retrained:
  - silver labels: 483 BAD, 1,611 OK;
  - silver CV AUC: CLIP-L 0.957, DINOv2 0.935.
- The pre-fix artifacts are kept:
  - `autolabel_data/v1_gate050/` (fused + students);
  - `models/v1_gate050/` (ONNX).
- `07` evaluates both versions.

Commands:

```bash
cd autolabel; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
nice -n 19 ../.venv/bin/python 04_fuse.py --vlm ../autolabel_data/vlm_gemma4e4b_snapshot_20261021.jsonl --scored-only --hand-thr 0.95
nice -n 19 ../.venv/bin/python 05_silver_train.py --emb clip_l14    --model logreg --allow-partial
nice -n 19 ../.venv/bin/python 05_silver_train.py --emb dinov2_base --model logreg --allow-partial
nice -n 19 ../.venv/bin/python 07_ingest_eval.py      # ~80 s (2000 bootstraps + leave-one-out calibration)
```

**`07_ingest_eval.py` was rewritten.** It now computes:
- for every score: AUC, AP(BAD), recall of BAD at 5%/10% FPR, ECE and Brier;
- label-stratified bootstrap 95% CIs for all of them;
- leave-one-out Platt and isotonic calibration;
- a sweep of gate rules;
- the percentile rank of each BAD crop under each score.

Outputs are `gold_check/eval.md` and `eval.json`, and `models/calibration.json`. That file uses Platt on the CLIP-L student: `a=0.387`, `b=2.310`, `p_ok = sigmoid(a*logit(p)+b)`.

**Results** (point [95% CI]). Full tables are in `gold_check/eval.md` and the README.

| score | AUC (OK vs BAD) |
|---|---|
| Gemma | 0.69 [0.49–0.87] |
| fused teacher | 0.67 [0.46–0.86] |
| CLIP-L student | 0.66 [0.44–0.85] |
| DINOv2 student | 0.51 [0.31–0.71] |
| detector conf alone | 0.71 [0.53–0.87] |

- Recall of BAD at 10% FPR is about 0.34 [0.0–0.7] for the best scores.
- With 9 BAD, none of the differences is significant.
- The silver CV AUC of 0.96 was only agreement with the teacher.
- An exploratory Gemma+detector rank average reached 0.72 (selected on gold, so optimistic). Dropping CLIP zero-shot from the fusion is a candidate change.

**Failure modes:**
- small crops (crop size alone gives AUC 0.65);
- subtle defects on grips and fists (#20, #38, #118, #147), which every score misses;
- stylized, non-human, robotic and occluded hands flagged as BAD.

**Next steps:**
1. When `ALL DONE`: `04_fuse.py` (full file, default `--hand-thr 0.95`), `05` ×2, `07`. Do not re-run `06`.
2. Build a stronger teacher: upscale crops from the full-resolution originals, use a larger VLM or several prompts, and consider dropping CLIP zero-shot from the fusion.
3. Run a second gold round of ~100 crops enriched with the lowest student scores (active learning) to get more than 9 positives.
4. Optional: add HAD positives.

## HADM-L evaluated as a teacher (2026-10-21 ~21:45 MSK)

Scripts are in `autolabel/09_hadm_kaggle/`; its `README.md` covers setup and porting. Data is in `autolabel_data/hadm/`: `input/`, `out_gold/` (kernel output + log) and `hadm_gold_scores.jsonl`. Results are in `gold_check/eval_hadm.{md,json}`.

- **Kaggle** (account `artem4767`, everything private):
  - dataset `artem4767/hand-defects-hadm-gold-input`: 145 source images, `boxes.json`, `hadm_infer.py`;
  - kernel `artem4767/hand-defects-hadm-gold`: T4, internet on.
  - One run, ~6.5 GPU-min: weights download 84 s, inference 1.61 s/image.
  - The kaggle CLI lives in `/workspace/.kaggle_venv`. Auth is `KAGGLE_API_TOKEN=$KAGGLE_KEY KAGGLE_USERNAME=artem4767`, and no token is stored in the repo.
- **Stack:** official HADM code @ `d09f667` on Kaggle's torch 2.11/cu128, not compiled. xformers attention is replaced by torch SDPA, plus small compatibility shims. The checkpoint loaded with no missing keys.
- **Result on the 107 gold hands** (9 BAD):
  - HADM-L alone: AUC 0.65–0.66 [~0.38–0.87], so no better than detector conf 0.71, Gemma 0.69 or the CLIP-L student 0.66. CountHallu got ≤ 0.64.
  - It is complementary: it catches the subtle grip defects #20, #30, #38 and partly #147, and misses the blurry gross ones #1, #8, #10.
  - Rank average HADM + Gemma + det_conf: **0.79 [0.65–0.91]**, chosen on gold, so optimistic. Gemma + det alone gives 0.72.
- **Next:** if the fusion is adopted, score all 2,500 source images, ~70 T4-min. Copy `kernel_gold` with a new input dataset, then add HADM as a third component in `04_fuse.py`.
  Then a second gold round with more BAD examples to confirm.

## Full-set HADM run + v3 teacher + gold v2 prepared (2026-10-21 ~22:40 MSK)

- **HADM-L on all crops:**
  - Input: 2,500 source images, of which 2,491 are Civitai originals at 1024 px long side and 9 are local 512-px fallbacks (CDN 404/522). Built with `09_hadm_kaggle/01_build_input.py --all` → `autolabel_data/hadm/input_all/`.
  - Uploaded as the private dataset `artem4767/hand-defects-hadm-all-input`, 605 MB.
  - Kernel `artem4767/hand-defects-hadm-all` (private), `09_hadm_kaggle/kernel_all/`: 2 shards, one per T4, the second starting after the first has loaded weights.
  - Session 34.5 min: weights 137 s, inference 31.6 min at 1.45 s/image per GPU.
  - Output is in `autolabel_data/hadm/out_all/`. The scores on the gold images are identical to the gold run.
  - `09_hadm_kaggle/04_crop_scores.py` → `autolabel_data/hadm_hand.jsonl` (`hadm_hand_iod50` = rule "box ≥ 50% inside crop").
- **Gemma:** `ALL DONE` 21:11 MSK, 5,604 lines, 0 errors.
- **Archived before overwriting:** `autolabel_data/v2_gate095_snap4325/` (fused, students, eval) and `models/v2_gate095_snap4325/` (ONNX + calibration).
- **Commands:**

```bash
cd autolabel
../.venv/bin/python 04_fuse.py --teacher hadm_gemma_det --hand-thr 0.95      # full vlm_gemma4e4b.jsonl; is_hand 4,314/5,604
../.venv/bin/python 10_goldset_v2.py --replace 138167122_0,144055360_0,137639480_1,144564484_2,135336233_0   # BEFORE training
../.venv/bin/python 05_silver_train.py --emb clip_l14 --model logreg         # excludes gold v1 + v2 + study-40
../.venv/bin/python 05_silver_train.py --emb dinov2_base --model logreg
../.venv/bin/python 07_ingest_eval.py                                         # gold v1 -> gold_check/eval.md, models/calibration.json
```

- **Silver labels:** pool 4,075 → 612 BAD (teacher ≤ 0.377) and 2,038 OK (≥ 0.536).
  Silver CV AUC: CLIP-L 0.895, DINOv2 0.874.
- **Gold v1** (optimistic for v3, because the teacher was chosen on it):
  - v3 teacher 0.792 [0.66–0.91];
  - CLIP-L v3 student 0.669 [0.44–0.85], versus 0.664 for v2;
  - DINOv2 v3 0.564.
  - Platt for CLIP-L v3: a=0.420, b=2.400.
- **Gold v2** (`gold_check_v2/`, not sent yet):
  - 100 crops on `sheet_01..04.jpg`. Strata: `low_teacher` 60 from the lowest 10% of the v3 teacher, `rest_q1..q4` 8 each, `gate_not_hand` 8.
  - Weights = stratum size / picks over the population of all crops minus gold v1 and study-40.
  - SFW check: all 4 sheets inspected. 5 crops were replaced, including old #31, #67, #72, #82 and #84: possible bare skin or torso, and hands on the chest area.
    The replacements were re-checked. The swaps are listed in `manifest.json`.
  - When the answers come, put them in `gold_check_v2/answers.txt` (same format as v1) and run:
    - `07_ingest_eval.py --gold-dir gold_check_v2` (honest);
    - `07_ingest_eval.py --gold-dir gold_check,gold_check_v2 [--write-calibration]` → `gold_check_combined/`.
  - Tested end to end on fake answers; those outputs were deleted.

## Gold v2 answered and evaluated (2026-10-21 ~22:55 MSK)

- **Answers** are in `gold_check_v2/answers.txt`, verbatim: bad 1 28 30 37 55 56 94 99; nohand 8 12 18 19 33 34 35 79 86 98 100; 14 skips.
  That gives 67 OK, 8 BAD, 11 nohand, 14 skip, so 75 gold hands. Nothing was renumbered or rebuilt.
- **Commands:**
  - `07_ingest_eval.py --gold-dir gold_check_v2` → `gold_check_v2/eval.md` + `eval.json` (honest);
  - `11_gold_v2_analysis.py` → `gold_check_v2/analysis.md` + `analysis.json` + `bad_and_false_alarms.jpg`: strata hit rate, v2 teacher on the 52 covered hands, gate, BAD ranks, false alarms;
  - `07_ingest_eval.py --gold-dir gold_check,gold_check_v2 --write-calibration` → `gold_check_combined/` and `models/calibration.json`.
- **Gold v2 AUC (weighted):**
  - HADM alone 0.763 [0.51–0.96];
  - v3 teacher 0.648 [0.43–0.85];
  - CLIP-L v3 student 0.585, v2 student 0.519;
  - det conf 0.538, Gemma 0.516;
  - DINOv2 v3 0.505;
  - v2 teacher 0.555, on its 52 covered hands only.
- **The 0.79 on gold v1 did not hold.**
- **Enrichment:** the low_teacher stratum had 4 BAD out of 43 judged hands (9.3%). The estimated population prevalence is 12.2%.
  In the rest strata, q1 had 2/7, q2 1/8, q3 1/8 and q4 0/8.
  Inside the low stratum, the v3 teacher has AUC 0.72 and HADM 0.80, while Gemma (0.37) and the CLIP-L student (0.29) are inverted.
- **Pooled v1 + v2** (182 hands, 17 BAD): HADM 0.717, v3 teacher 0.714, CLIP-L v3 0.605.
  Platt was refit for the CLIP-L v3 student: a=0.268, b=2.165, LOO ECE 0.336 → 0.038. The ONNX models are unchanged, because Platt is applied outside the model.
- **Gate on v2:** 7 of the 8 gate_not_hand crops are nohand; #70 is an OK hand with p_hand 0.79. 4 nohand crops passed with p_hand ≥ 0.98: #8, #79, #98, #100.
  Weighted nohand share among passed crops is 0.7%.
- **Failure modes:**
  1. False alarms on unusual-but-fine hands, now also from HADM: skeletal (#88, #95), robotic or chrome (#92, #68), glove fist (#23), stylized anime heart-hands (#97), blurry or noisy small hands (#16, #50).
  2. Missed BAD in grips and holds: bowl #1, guitar #99, anime desk hand #94. Gemma and the detector rank these as clean and pull the fused score back up, even where HADM flags them (#1, #99).
  3. Gemma's p_ok is near chance on this round.
- **Next:** use HADM alone, or HADM-weighted, as the teacher, and drop Gemma p_ok from the quality score; keep Gemma only for the hand gate.
  Then fine-tune a crop classifier on the HADM-based silver labels, or train a hand_bad detector on AbHuman/HAD plus HADM silver labels, on Kaggle.
  Before claiming any gain, run gold v3 with ~100 crops sampled by HADM score.

## Check the VLM job

```bash
cd /workspace/hand_defects
cat autolabel_data/vlm_job.pid                       # PID of the restart loop (run_vlm.sh)
pgrep -af 03_vlm_score.py                            # worker PID (python)
ps -o pid,stat,etime,cmd -p $(pgrep -f 03_vlm_score.py)   # STAT 'T' = paused, 'R'/'S' = running
grep -E "start|progress|resumed|restarting|ALL DONE" autolabel_data/vlm_gemma4e4b.log | tail -5
wc -l < autolabel_data/vlm_gemma4e4b.jsonl           # crops done (target 5604)
grep -c '"error"' autolabel_data/vlm_gemma4e4b.jsonl # failed crops (not retried automatically, see below)
```

The log also contains llama.cpp/mtmd noise. Filter it with the `grep` above.

## Restart / resume (safe at any time)

The job is resumable: results are appended to `autolabel_data/vlm_gemma4e4b.jsonl` one line per crop, and on start every key already present is skipped. `run_vlm.sh` restarts a crashed worker up to 20 times.

```bash
cd /workspace/hand_defects
# if it is paused (STAT T) and step 2 is no longer running:
kill -CONT $(pgrep -f 03_vlm_score.py)
# if nothing is running (e.g. after a reboot):
setsid nohup autolabel/run_vlm.sh >/dev/null 2>&1 < /dev/null &
# stop it cleanly:
kill $(cat autolabel_data/vlm_job.pid); pkill -f 03_vlm_score.py
```

The worker runs with `VLM_THREADS=6`. 8 threads measured no faster: 4.7–5.9 vs ~5.0 s/crop. To change threads, restart it. Nothing is lost, only the current crop is recomputed:

```bash
kill $(cat autolabel_data/vlm_job.pid); pkill -f 03_vlm_score.py; sleep 3
VLM_THREADS=8 setsid nohup env VLM_THREADS=8 autolabel/run_vlm.sh >/dev/null 2>&1 < /dev/null &
```

Crops whose line contains `"error"` are counted as done. To retry them, delete those lines (`grep -v '"error"' f > f.tmp && mv f.tmp f`) and restart.

If step 2 was interrupted, rerun it: `EMB_THREADS=8 .venv/bin/python autolabel/02_cheap_scores.py dino l14`. Finished `.npy` files are skipped. **Pause the VLM worker first.**

## After the VLM job finishes (`ALL DONE` in the log, 5,604 lines)

```bash
cd /workspace/hand_defects/autolabel
../.venv/bin/python 04_fuse.py                      # -> autolabel_data/fused.jsonl (teacher_ok, is_hand)
../.venv/bin/python 06_goldset.py                   # FIRST build the gold set (stratified by teacher_ok), so training can exclude it
#   -> autolabel_data/gold_check/sheet_01..06.jpg (25 numbered crops each), manifest.json, answers_TEMPLATE.txt
#   !! look at every sheet before showing it to the user (SFW check); if a crop is unsuitable, rerun with another --seed
../.venv/bin/python 05_silver_train.py --emb clip_l14 --model logreg
../.venv/bin/python 05_silver_train.py --emb dinov2_base --model logreg
../.venv/bin/python 05_silver_train.py --emb clip_l14 --model mlp       # optional
#   -> autolabel_data/student_<emb>_<model>.jsonl (P(ok) for all crops), models/hand_ok_<emb>_<model>.onnx
```

Then send the 6 sheets to the user. He answers with the numbers of the bad crops, e.g. `bad: 4 17 33; nohand: 2 40; skip: 9`. Everything not listed counts as OK. Put the answer into `autolabel_data/gold_check/answers.txt` (format in `answers_TEMPLATE.txt`, one label per line) and run:

```bash
../.venv/bin/python 07_ingest_eval.py                # or --main clip_l14_logreg / dinov2_base_logreg
#   -> gold_check/eval.md + eval.json, models/calibration.json (Platt + isotonic for the main score)
```

`eval.md` contains:
- AUC for every score (teacher, Gemma, CLIP, detector conf, each student);
- 2-fold calibrate-on-half / test-on-half numbers: AUC, recall on BAD at 5% and 10% false-alarm rate (threshold picked on the calibration half), and ECE/Brier for raw, Platt and isotonic probabilities;
- precision/recall of the "not a hand" gate.

All metrics are re-weighted by the stratified sampling weights.

## Next steps after the gold check

1. Pick the student with the best gold AUC, apply `models/calibration.json`, and write the numbers into README (honest: "user-checked 150-crop set").
2. Optional upgrade: add HAD (Human Artifact Dataset) hand-artifact crops as extra BAD examples. See the feasibility report: 8.8 GB, no license file, so research use only.
3. Stage 2 of the README: train YOLO / two-stage detector+classifier on Kaggle using `is_hand` + silver labels, then the ComfyUI node.

## Design notes

- **Prompt** (identical to the study): "Look at this cropped image region from an AI-generated picture. Judge only the hand(s). Is the hand anatomically correct (normal number of fingers, no fused/extra/missing fingers, no melted or broken shape)? Answer with exactly one word: Yes if the hand looks correct, No if it is malformed, None if there is no human hand."
- **Scores taken from it:**
  - p_ok = P(Yes)/(P(Yes)+P(No));
  - p_hand = 1 − P(None)/(P(Yes)+P(No)+P(None)).
  One call gives both, so no second "is this a hand" question is needed. In the study, non-hands had P(None) ≥ 0.89 and hands ≤ 0.04.
- **Handler:** generic `MTMDChatHandler`. The Qwen-specific handler breaks first-token logprobs.
- **Fusion:** equal-weight mean of rank-normalised Gemma p_ok, detector confidence and CLIP p_ok. This was the best combination in the study: AUC 0.93, or 0.89 on the subset not mined with CLIP. There is no fitted logistic fusion, because 35 labelled hands would overfit it. Once the gold set exists, `07` reports each component separately, so the weights can be revisited.
- **Silver labels:** bottom 15% = BAD, top 50% = OK, the middle 35% is discarded. Only is_hand crops are used. Gold-check and study crops are excluded.
- **Dry run** (before the VLM, teacher = detector + CLIP only, CLIP-B/32 logreg):
  - CV agreement with silver labels: AUC 0.98;
  - on the 35 study hands: student 0.90 vs teacher 0.89.
  These are not honest metrics.

## Wrap-up: HADM scorer + ComfyUI node pack (2026-10-21 ~23:30 MSK)

- Final choice: **HADM-L is the hand-quality scorer**. Gemma is used only as the offline hand/not-hand gate (p_hand ≥ 0.95), and the CLIP-L student is kept as a light, clearly weaker fallback.
- `autolabel/12_hadm_calibration.py` → `models/hadm_calibration.json`: Platt on pooled gold (182 hands, 17 BAD), `p_bad = sigmoid(4.108·raw − 4.039)`, LOO ECE 0.013, Brier 0.086.
- `hand_defects/hadm_scorer.py`: Python API and CLI. Its `_hadm_backend.py`, `_geometry.py` and `_detector.py` are byte-identical copies of the node pack's files, which `tests/test_sync.py` checks.
- `comfyui_hand_fixer/`: a separate git repository ([ComfyUI-Hand-Fixer](https://github.com/tablesizedogo/ComfyUI-Hand-Fixer)), ignored by this one. It contains the ComfyUI node pack, an API-format example workflow, 14 CPU tests (mocks plus a real HADM build with random weights plus the real YOLO), and the Kaggle E2E script.
- E2E test: private Kaggle kernel `artem4767/hand-fixer-comfyui-e2e`, T4, 11.3 min session. Headless ComfyUI with DreamShaper 8 inpainting, HADM-L and hand_yolov8s on 5 SFW images with gold-BAD hands.
  - 5/5 workflows succeeded.
  - 3 hands were flagged, and Best-of-4 lowered P(bad) on all 3 (0.29→0.16, 0.23→0.19, 0.23→0.20).
  - By eye: 1 clear improvement, 1 ambiguous, and 1 where the generic prompt removed the held revolver.
  - sha256 values from the kernel log:
    - HADM-L_0249999.pth `3a495952778fbb9e67024d1fc48e49a5be55a21a58c608741ccb02c1dfd407f3`
    - hand_yolov8s.pt `70b540063fbc385736d8258970744a4afbc4cbf7932134bae3b24cdadeadec06`
    - DreamShaper_8_INPAINTING `ac84e05d5039d7a8aea3ced335ac7c56da6d3c6816465651f928b137c3a7b24a`
- The old README moved to `docs/DETAILED_LOG.md`, and the new README is the short summary.
- Total Kaggle GPU time for the project is about 53 min.
