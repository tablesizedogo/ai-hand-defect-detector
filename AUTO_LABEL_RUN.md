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
| 2. cheap scores + embeddings | `autolabel/02_cheap_scores.py` | CLIP-B/32 zero-shot (`clip_b32_zs.jsonl`) + `emb_clip_b32.npy` **done**. `emb_dinov2_base.npy`, then `emb_clip_l14.npy`, **running** (log `autolabel_data/cheap_scores2.log`). The detector confidence is already in the manifest (`det_conf`) |
| 3. Gemma-4-E4B Yes/No/None | `autolabel/03_vlm_score.py` via `autolabel/run_vlm.sh` | **running, detached**. It was paused (SIGSTOP) while step 2 used all 8 cores. A detached watcher resumes it (SIGCONT) when step 2 exits. ~5 s/crop alone → **ETA ≈ 7.5–8 h after resume** (resume expected ≈ 15:10 MSK when CLIP-L finishes → done ≈ 23:00 MSK 2026-10-20). Worker PID 331828, loop PID 331825 |
| 4. fusion + hand gate | `autolabel/04_fuse.py` | ready, tested (dry run) |
| 5. silver labels + student + ONNX | `autolabel/05_silver_train.py` | ready, tested (dry run, CLIP-B/32: logreg + MLP, ONNX parity 2e-7) |
| 6. gold-check set + contact sheets | `autolabel/06_goldset.py` | ready, tested (dry run) |
| 7. ingest answers → calibration + eval | `autolabel/07_ingest_eval.py` | ready, tested with fake answers |

Why the pause: running torch embeddings next to llama.cpp made Gemma go from 5 s to 25–40 s per crop because of CPU oversubscription. **Never run heavy CPU jobs while step 3 runs.** If you must, pause the worker first: `kill -STOP <worker pid>`, and later `kill -CONT <worker pid>`.

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

The worker uses 6 threads (it was started with `VLM_THREADS=6`). After step 2 finishes, you can restart it with all 8 threads for ~15–20% more speed. Nothing is lost, only the current crop is recomputed:

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
