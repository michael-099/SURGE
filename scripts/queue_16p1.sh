#!/usr/bin/env bash
# Section 16 (research/log.md), P1: decoder state across windows. Step 0 (test-time stream buffer on runs/vitg_s0,
# descriptive), then P1a (scripts/train_multi.py --pairs, decoders R, Rall, S) for seed 0; seeds 1 and 2 only if S of
# seed 0 passes the gate (holdout CC >= 0.5836). Then the summary (CPU), the descriptive per-position probes of the
# trained decoders, and the summary again. One GPU job at a time; stops if SAFETY_STOP.txt (repo root) exists.
# Resumable (finished runs and outputs are skipped).
# Usage: bash scripts/queue_16p1.sh > runs/p1/queue.log 2>&1
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
L=runs/p1
O=runs/review/p1
mkdir -p $L $O/probe
stop() { if [ -f SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; fi; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
summary() { run env CUDA_VISIBLE_DEVICES="" $PY -u scripts/inv_summary.py --part p1 > $L/summary.log 2>&1; }
PE="$PY -u scripts/probe_eval.py --split holdout"
echo "$(date -Is) queue 16p1 starts"
# ---- step 0: the existing seed-0 model with and without the buffer (all-16 tiling, per-position CC)
[ -f $O/step0_all16.json ] || run $PE --run runs/vitg_s0 --tiling all16 --positions \
    --out $O/step0_all16.json > $L/step0_all16.log 2>&1
[ -f $O/step0_all16_stream.json ] || run $PE --run runs/vitg_s0 --tiling all16 --stream --positions \
    --out $O/step0_all16_stream.json > $L/step0_all16_stream.log 2>&1
summary
# ---- P1a
G="model.backbone=vjepa2_g train.epochs=3 train.warmup_epochs=0.25 data.size=[256,256]"
TM="$PY -u scripts/train_multi.py --config configs/main_b1.yaml --set"
train() { [ -f runs/p1_s$1/DONE ] || run $TM run_dir=runs/p1_s$1 train.seed=$1 $G --pairs --variants R,Rall,S \
    > $L/p1_s$1.log 2>&1; }
train 0
if $PY -I -c "import json, sys; sys.exit(0 if json.load(open('runs/p1_s0/S/best.json'))['mean']['cc'] >= 0.5836 else 1)"
then
  echo "$(date -Is) gate passed (S of seed 0 >= 0.5836)"
  train 1
  train 2
else
  echo "$(date -Is) gate not passed (S of seed 0 < 0.5836): seeds 1-2 not run"
fi
summary
# ---- descriptive: per-position CC on the all-16 tiling of the trained decoders (best epochs)
for s in 0 1 2; do
  [ -f runs/p1_s$s/DONE ] || continue
  P="$PE --run runs/p1_s$s --tiling all16 --positions"
  [ -f $O/probe/p1_s${s}_S_stream.json ] || run $P --variant S --stream \
      --out $O/probe/p1_s${s}_S_stream.json > $L/probe_s${s}_S_stream.log 2>&1
  [ -f $O/probe/p1_s${s}_S_nostream.json ] || run $P --variant S \
      --out $O/probe/p1_s${s}_S_nostream.json > $L/probe_s${s}_S_nostream.log 2>&1
  [ -f $O/probe/p1_s${s}_Rall.json ] || run $P --variant Rall \
      --out $O/probe/p1_s${s}_Rall.json > $L/probe_s${s}_Rall.log 2>&1
  [ -f $O/probe/p1_s${s}_R_all16.json ] || run $P --variant R \
      --out $O/probe/p1_s${s}_R_all16.json > $L/probe_s${s}_R_all16.log 2>&1
done
summary
echo "$(date -Is) queue 16p1 done"
