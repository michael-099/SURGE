#!/usr/bin/env bash
# research/log.md section 12A and 12B (existing models only). One GPU job at a time. Resumable.
# Stops if SAFETY_STOP.txt exists or a step fails. Usage: bash scripts/queue_review_a.sh > runs/review/queue_a.log 2>&1
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
sig() { $PY -c "import json;print(json.load(open('$1'))['best_sigma'])"; }
cmp() { [ -f $3 ] || $PY scripts/compare.py $1 $2 --out ${3%.txt}.json > $3; }
O=runs/review; mkdir -p $O
SAL=runs/metric_check/salfom_val_all.json

# ------------------------------------------------------------------ 12A seeds 1 and 2, singly, on DHF1K val
for s in 1 2; do
  [ -f $O/val_single_l_s$s.json ] || run $PY -u scripts/evaluate.py --run runs/main_b1_s$s --split val --metrics all \
      --sigma 12 --out $O/val_single_l_s$s.json > $O/val_single_l_s$s.log 2>&1
  cmp $O/val_single_l_s$s.json $SAL $O/cmp_single_l_s${s}_vs_salfom.txt
done
for s in 1 2; do
  [ -f $O/val_single_g_s$s.json ] || run $PY -u scripts/evaluate.py --run runs/vitg_s$s --split val --metrics all \
      --sigma 8 --out $O/val_single_g_s$s.json > $O/val_single_g_s$s.log 2>&1
  cmp $O/val_single_g_s$s.json $SAL $O/cmp_single_g_s${s}_vs_salfom.txt
done

# ------------------------------------------------------------------ 12B frozen UMT-L + SURGE-D decoder
[ -f $O/sigma_umt.json ] || run $PY -u scripts/tune_sigma.py --run runs/g1_umt_l --out $O/sigma_umt.json > $O/sigma_umt.log 2>&1
SU=$(sig $O/sigma_umt.json)
[ -f $O/val_umt.json ] || run $PY -u scripts/evaluate.py --run runs/g1_umt_l --split val --metrics all --sigma $SU \
    --out $O/val_umt.json > $O/val_umt.log 2>&1
cmp $O/val_umt.json $SAL $O/cmp_umt_vs_salfom.txt
cmp runs/final/val_single_s0.json $O/val_umt.json $O/cmp_vitl_s0_vs_umt.txt
echo "$(date -Is) queue A done"
