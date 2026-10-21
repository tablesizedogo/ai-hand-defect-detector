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
../.venv/bin/python 04_fuse.py            # full vlm_gemma4e4b.jsonl, no snapshot
../.venv/bin/python 05_silver_train.py --emb clip_l14 --model logreg
../.venv/bin/python 05_silver_train.py --emb dinov2_base --model logreg
```

**Do NOT re-run `06_goldset.py` once the sheets have been sent.** It would renumber the set. The existing `gold_check/manifest.json` is kept, and `05` keeps excluding those crops from training.

The gold sampling weights describe the snapshot population (3,519 hands). That is fine because the scored subset is a large, style-balanced sample of the full set. Re-run `07_ingest_eval.py` after retraining to get the gold metrics for the new students.

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
