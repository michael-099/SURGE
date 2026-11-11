#!/usr/bin/env bash
# research/log.md section 12D addendum (ViT-L, written 2026-11-11 08:33): the offline ViT-L models rescored on the
# every-10th DHF1K val frames of 12D, then causal - offline and offline - SalFoM on those frames.
# One GPU job at a time. Resumable. Stops if SAFETY_STOP.txt exists or a step fails.
# Usage: bash scripts/queue_review_c.sh > runs/review/queue_c.log 2>&1
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
cmp() { [ -f $3 ] || $PY scripts/compare.py $1 $2 --out ${3%.txt}.json > $3; }
O=runs/review

# offline ViT-L with the sigma of section 5 (single seed 0: 12, three-seed ensemble: 8), no new tuning
echo "$(date -Is) offline ViT-L, seed 0"
[ -f $O/offline_l_val_step10_s0.json ] || run $PY -u scripts/evaluate.py --run runs/main_b1_s0 --split val \
    --metrics all --sigma 12 --frame-step 10 --out $O/offline_l_val_step10_s0.json > $O/offline_l_val_step10_s0.log 2>&1
[ -f $O/offline_l_val_step10_s0.json ] || { echo "$(date -Is) no offline_l_val_step10_s0.json; queue stops"; exit 3; }
echo "$(date -Is) offline ViT-L, ensemble"
[ -f $O/offline_l_val_step10_ens3.json ] || run $PY -u scripts/evaluate.py --run runs/main_b1_s0 --run runs/main_b1_s1 \
    --run runs/main_b1_s2 --split val --metrics all --sigma 8 --frame-step 10 --out $O/offline_l_val_step10_ens3.json \
    > $O/offline_l_val_step10_ens3.log 2>&1
[ -f $O/offline_l_val_step10_ens3.json ] || { echo "$(date -Is) no offline_l_val_step10_ens3.json; queue stops"; exit 3; }

# paired bootstraps (CPU)
cmp $O/causal_l_val_s0.json $O/offline_l_val_step10_s0.json $O/cmp_causal_vs_offline_l_s0.txt
cmp $O/causal_l_val_ens3.json $O/offline_l_val_step10_ens3.json $O/cmp_causal_vs_offline_l_ens3.txt
for x in s0 ens3; do
  cmp $O/offline_l_val_step10_$x.json $O/salfom_val_step10_best.json $O/cmp_offline_l_${x}_step10_vs_salfom_best.txt
  cmp $O/offline_l_val_step10_$x.json $O/salfom_val_step10_sigma0.json $O/cmp_offline_l_${x}_step10_vs_salfom.txt
done
echo "$(date -Is) queue C done"
