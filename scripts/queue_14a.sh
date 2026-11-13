#!/usr/bin/env bash
# Section 14A pilots (research/log.md): SURGE-D ViT-g + a second frozen expert, seed 0, the ViT-g main recipe.
# Stops before each run if SAFETY_STOP.txt exists. One GPU job at a time.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
for EXP in dinov2_l umt_l; do
  RUN=runs/xp_${EXP}_s0
  if [ -e SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP present, not starting $RUN"; exit 1; fi
  if [ -f $RUN/DONE ]; then echo "$(date -Is) $RUN already done"; continue; fi
  echo "$(date -Is) start $RUN"
  $PY -u scripts/train.py --config configs/main_b1.yaml --set run_dir=$RUN train.seed=0 train.epochs=3 \
      train.warmup_epochs=0.25 model.backbone=vjepa2_g "data.size=[256,256]" model.expert=$EXP > $RUN.log 2>&1
  echo "$(date -Is) end $RUN exit $?"
done
echo "$(date -Is) queue 14A done"
