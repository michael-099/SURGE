#!/usr/bin/env bash
# Section 14D (research/log.md): seeds 3-5 of SURGE-D ViT-g (reference) and ViT-g + UMT-L, then per-video holdout
# scores with the command of 12E. Stops before each job if SAFETY_STOP.txt exists in the repo root.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
stop() { if [ -e SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP present, stopping before $1"; exit 1; fi; }
for SEED in 3 4 5; do
  for KIND in ref umt; do
    if [ $KIND = ref ]; then RUN=runs/vitg_s$SEED; EXTRA=""; else RUN=runs/xp_umt_l_s$SEED; EXTRA="model.expert=umt_l"; fi
    stop $RUN
    if [ -f $RUN/DONE ]; then echo "$(date -Is) $RUN already done"; continue; fi
    echo "$(date -Is) start $RUN"
    $PY -u scripts/train.py --config configs/main_b1.yaml --set run_dir=$RUN train.seed=$SEED train.epochs=3 \
        train.warmup_epochs=0.25 model.backbone=vjepa2_g "data.size=[256,256]" $EXTRA > $RUN.log 2>&1
    echo "$(date -Is) end $RUN exit $?"
  done
done
O=runs/review/holdout; mkdir -p $O
for R in vitg_s3 vitg_s4 vitg_s5 xp_umt_l_s3 xp_umt_l_s4 xp_umt_l_s5; do
  stop "holdout $R"
  if [ -f $O/$R.json ]; then echo "$(date -Is) holdout $R exists"; continue; fi
  echo "$(date -Is) holdout $R"
  $PY -u scripts/evaluate.py --run runs/$R --split holdout --metrics cc,nss,sim,kld --frame-step 5 --sigma 0 \
      --out $O/$R.json > $O/$R.log 2>&1
  echo "$(date -Is) holdout $R exit $?"
done
echo "$(date -Is) queue 14D done"
