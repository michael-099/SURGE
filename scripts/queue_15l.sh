#!/usr/bin/env bash
# Section 15L (research/log.md): scene-cut hypothesis test. Dense windows with and without clamping to the frame's
# shot: DIEM fold-1 test with the fine-tuned ViT-g fold models (seeds 0-2), DHF1K holdout with runs/vitg_s0. Waits for
# queue 15t (one GPU job at a time); queue 15b waits for this one. Stops if SAFETY_STOP.txt exists. Resumable.
# Usage: bash scripts/queue_15l.sh > runs/inv/queue_15l.log 2>&1
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
O=runs/review/inv
L=runs/inv
stop() { if [ -f SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; fi; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
n=0
until grep -q "queue 15t done\|SAFETY_STOP present\|queue stops\|FAILED\|not starting" $L/queue_15t.log 2>/dev/null; do
  n=$((n+1)); if [ $n -gt 900 ]; then echo "$(date -Is) queue 15t not finished after 15 h; not starting"; exit 1; fi
  sleep 60
done
grep -q "queue 15t done" $L/queue_15t.log || { echo "$(date -Is) queue 15t did not finish cleanly; not starting"; exit 1; }
sleep 20
echo "$(date -Is) queue 15l starts"
P="$PY -u scripts/probe_eval.py"
for s in 0 1 2; do
  R=runs/g_stavis_diem_f1_s$s
  [ -f $O/l_diem_dense_s$s.json ] || run $P --run $R --dataset diem_f1 --split val --tiling dense --per-frame \
      --out $O/l_diem_dense_s$s.json > $L/l_diem_dense_s$s.log 2>&1
  [ -f $O/l_diem_shot_s$s.json ] || run $P --run $R --dataset diem_f1 --split val --tiling dense --shots diem --per-frame \
      --out $O/l_diem_shot_s$s.json > $L/l_diem_shot_s$s.log 2>&1
done
[ -f $O/l_dhf1k_shot_s0.json ] || run $P --run runs/vitg_s0 --split holdout --tiling dense --shots dhf1k --per-frame \
    --out $O/l_dhf1k_shot_s0.json > $L/l_dhf1k_shot_s0.log 2>&1
CUDA_VISIBLE_DEVICES="" $PY scripts/inv_cuts.py --out $O/summary_cuts.json | tee $L/summary_cuts.txt
echo "$(date -Is) queue 15l done"
