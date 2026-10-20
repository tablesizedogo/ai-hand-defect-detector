#!/bin/bash
# Detached, self-restarting launcher for the resumable VLM job (step 3).
# Usage: setsid nohup autolabel/run_vlm.sh >/dev/null 2>&1 &
cd /workspace/hand_defects
PY=/workspace/rag_civitai/search/.venv/bin/python   # venv with llama-cpp-python 0.3.36
LOG=autolabel_data/vlm_gemma4e4b.log
echo $$ > autolabel_data/vlm_job.pid
for attempt in $(seq 1 20); do
  $PY autolabel/03_vlm_score.py >> $LOG 2>&1
  if tail -n 3 $LOG | grep -q "ALL DONE"; then break; fi
  echo "$(date '+%F %T') worker exited (attempt $attempt), restarting in 30 s" >> $LOG
  sleep 30
done
rm -f autolabel_data/vlm_job.pid
