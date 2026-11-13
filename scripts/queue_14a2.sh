#!/usr/bin/env bash
# Section 14A, second stage (research/log.md): seeds 1 and 2 of both expert candidates, then per-video holdout
# scores of the nine runs of the three-seed test (the command of 12E: every fifth frame, sigma 0).
# Stops before each job if SAFETY_STOP.txt exists. One GPU job at a time.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
stop() { if [ -e SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP present, stopping before $1"; exit 1; fi; }
for SEED in 1 2; do
  for EXP in dinov2_l umt_l; do
    RUN=runs/xp_${EXP}_s${SEED}
    stop $RUN
    if [ -f $RUN/DONE ]; then echo "$(date -Is) $RUN already done"; continue; fi
    echo "$(date -Is) start $RUN"
    $PY -u scripts/train.py --config configs/main_b1.yaml --set run_dir=$RUN train.seed=$SEED train.epochs=3 \
        train.warmup_epochs=0.25 model.backbone=vjepa2_g "data.size=[256,256]" model.expert=$EXP > $RUN.log 2>&1
    echo "$(date -Is) end $RUN exit $?"
  done
done
O=runs/review/holdout; mkdir -p $O
for R in vitg_s0 vitg_s1 vitg_s2 xp_dinov2_l_s0 xp_dinov2_l_s1 xp_dinov2_l_s2 xp_umt_l_s0 xp_umt_l_s1 xp_umt_l_s2; do
  stop "holdout $R"
  if [ -f $O/$R.json ]; then echo "$(date -Is) holdout $R exists"; continue; fi
  echo "$(date -Is) holdout $R"
  $PY -u scripts/evaluate.py --run runs/$R --split holdout --metrics cc,nss,sim,kld --frame-step 5 --sigma 0 \
      --out $O/$R.json > $O/$R.log 2>&1
  echo "$(date -Is) holdout $R exit $?"
done
echo "$(date -Is) queue 14A2 done"
