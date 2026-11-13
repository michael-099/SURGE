#!/usr/bin/env bash
# Section 14E (research/log.md): predictability-gated decoder. Waits for queue 14D to finish (one GPU job at a time),
# then: GPU smoke test, seed 0, seeds 1-2 if the seed-0 holdout CC >= 0.5896, and per-video holdout scores with the
# matched, 180-degree-rotated and zero error maps (the command of 12E). Stops if SAFETY_STOP.txt (repo root) exists.
set -u
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe
stop() { if [ -e SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP present, stopping before $1"; exit 1; fi; }
n=0
until grep -q "queue 14D done\|SAFETY_STOP present" runs/queue_14d.log 2>/dev/null; do
  n=$((n+1)); if [ $n -gt 900 ]; then echo "$(date -Is) queue 14D not finished after 15 h; not starting"; exit 1; fi
  sleep 60
done
echo "$(date -Is) queue 14D finished; starting 14E"
PG="model.backbone=vjepa2_g data.size=[256,256] model.pgate=true"
stop smoke
if [ ! -f runs/smoke_pgate/DONE ]; then
  $PY -u scripts/train.py --config configs/smoke.yaml --set run_dir=runs/smoke_pgate $PG > runs/smoke_pgate.log 2>&1
  [ -f runs/smoke_pgate/DONE ] || { echo "$(date -Is) smoke test failed (see runs/smoke_pgate.log)"; exit 1; }
fi
echo "$(date -Is) smoke test passed"
train_seed() {
  local S=$1 RUN=runs/pg_g_s$1
  stop $RUN
  if [ -f $RUN/DONE ]; then echo "$(date -Is) $RUN already done"; return 0; fi
  echo "$(date -Is) start $RUN"
  $PY -u scripts/train.py --config configs/main_b1.yaml --set run_dir=$RUN train.seed=$S train.epochs=3 \
      train.warmup_epochs=0.25 $PG > $RUN.log 2>&1
  echo "$(date -Is) end $RUN exit $?"
  [ -f $RUN/DONE ] || { echo "$(date -Is) no DONE in $RUN; queue stops"; exit 1; }
}
train_seed 0
BEST=$($PY -c "import json;print(json.load(open('runs/pg_g_s0/DONE'))['best_holdout_cc'])")
SEEDS="0"
if $PY -c "import sys; sys.exit(0 if $BEST >= 0.5896 else 1)"; then
  echo "$(date -Is) seed 0 holdout CC $BEST >= 0.5896: seeds 1 and 2"
  train_seed 1; train_seed 2; SEEDS="0 1 2"
else
  echo "$(date -Is) seed 0 holdout CC $BEST < 0.5896: not continued (rule 14E)"
fi
O=runs/review/holdout; mkdir -p $O
for S in $SEEDS; do
  for CTL in matched rot180 zero; do
    R=pg_g_s$S; F=$R; EXTRA=""
    [ $CTL = rot180 ] && { F=${R}_rot; EXTRA="--pgate-control rot180"; }
    [ $CTL = zero ] && { F=${R}_zero; EXTRA="--pgate-control zero"; }
    stop "holdout $F"
    [ -f $O/$F.json ] && { echo "$(date -Is) holdout $F exists"; continue; }
    echo "$(date -Is) holdout $F"
    $PY -u scripts/evaluate.py --run runs/$R --split holdout --metrics cc,nss,sim,kld --frame-step 5 --sigma 0 \
        $EXTRA --out $O/$F.json > $O/$F.log 2>&1
    echo "$(date -Is) holdout $F exit $?"
  done
done
echo "$(date -Is) queue 14E done"
