#!/usr/bin/env bash
# Section 15C2 (research/log.md): which part of V3 carries its effect, the temporal convolutions or the clip-pooled
# GroupNorm statistics? Variants R, V3, V3a, V3b on one frozen-encoder stream (scripts/train_multi.py), seeds 0-2,
# then the summary (CPU). One GPU job at a time: starts only once queue 15b's last output
# (runs/review/inv/a3_cost.json) exists and queue 15b's process (its PID, the first argument) has ended.
# Stops if SAFETY_STOP.txt (repo root) exists. Resumable (finished runs are skipped).
# Usage: bash scripts/queue_15c.sh <pid of queue 15b> > runs/inv/queue_15c.log 2>&1
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
L=runs/inv
P15B=${1:-}
LAST=runs/review/inv/a3_cost.json
stop() { if [ -f SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; fi; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
alive() { [ -n "$P15B" ] && ps -p "$P15B" >/dev/null 2>&1; }
echo "$(date -Is) queue 15c waiting for queue 15b (pid ${P15B:-none})"
n=0
until [ -f $LAST ] && ! alive; do
  stop
  if [ -n "$P15B" ] && ! alive && [ ! -f $LAST ]; then
    echo "$(date -Is) queue 15b ended without $LAST; not starting"; exit 1
  fi
  n=$((n+1)); if [ $n -gt 2400 ]; then echo "$(date -Is) queue 15b not finished after 40 h; not starting"; exit 1; fi
  sleep 60
done
sleep 30
echo "$(date -Is) queue 15b done; starting queue 15c"
G="model.backbone=vjepa2_g train.epochs=3 train.warmup_epochs=0.25 data.size=[256,256]"
TM="$PY -u scripts/train_multi.py --config configs/main_b1.yaml --set"
for s in 0 1 2; do
  [ -f runs/inv_c2_s$s/DONE ] || run $TM run_dir=runs/inv_c2_s$s train.seed=$s $G --variants R,V3,V3a,V3b \
      > $L/inv_c2_s$s.log 2>&1
done
run env CUDA_VISIBLE_DEVICES="" $PY -u scripts/inv_summary.py --part v3split > $L/inv_c2_summary.log 2>&1
echo "$(date -Is) queue 15c done"
