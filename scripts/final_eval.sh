#!/usr/bin/env bash
# Final steps (research/log.md section 4), one GPU job at a time, after the main queue:
#  1. blur sigma tuned on the DHF1K holdout for the main model (seed 0) and for the 3-seed ensemble
#  2. DHF1K val (official 100 videos), scored ONCE: seed 0 single, 3-seed ensemble (all six metrics)
#  3. paired bootstrap vs SalFoM's val maps (same evaluator)
#  4. UCF Sports: fine-tune from the main DHF1K model (seed 0), select on the UCF holdout, score the 47 test videos once
# Usage: MAIN=main_b1 bash scripts/final_eval.sh      (MAIN=main_m if LoRA passed its rule)
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
MAIN="${MAIN:-main_b1}"
OUT=runs/final; mkdir -p $OUT
until grep -qE "chain finished|chain stops" runs/chain_main.log 2>/dev/null; do sleep 30; done
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; final_eval stops"; exit 2; }; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }

run $PY -u scripts/tune_sigma.py --run runs/${MAIN}_s0 --out $OUT/sigma_single.json > $OUT/sigma_single.log 2>&1
run $PY -u scripts/tune_sigma.py --run runs/${MAIN}_s0 --run runs/${MAIN}_s1 --run runs/${MAIN}_s2 --out $OUT/sigma_ens3.json > $OUT/sigma_ens3.log 2>&1
S1=$($PY -c "import json;print(json.load(open('$OUT/sigma_single.json'))['best_sigma'])")
S3=$($PY -c "import json;print(json.load(open('$OUT/sigma_ens3.json'))['best_sigma'])")
echo "$(date -Is) sigma single=$S1 ens3=$S3"

run $PY -u scripts/evaluate.py --run runs/${MAIN}_s0 --split val --metrics all --sigma $S1 --out $OUT/val_single_s0.json > $OUT/val_single_s0.log 2>&1
run $PY -u scripts/evaluate.py --run runs/${MAIN}_s0 --run runs/${MAIN}_s1 --run runs/${MAIN}_s2 --split val --metrics all --sigma $S3 \
    --out $OUT/val_ens3.json > $OUT/val_ens3.log 2>&1
$PY scripts/compare.py $OUT/val_single_s0.json runs/metric_check/salfom_val_all.json --out $OUT/cmp_single_vs_salfom.json > $OUT/cmp_single_vs_salfom.txt
$PY scripts/compare.py $OUT/val_ens3.json runs/metric_check/salfom_val_all.json --out $OUT/cmp_ens3_vs_salfom.json > $OUT/cmp_ens3_vs_salfom.txt
echo "$(date -Is) DHF1K val done"

# UCF Sports
STAGE=1; [ "$MAIN" = "main_m" ] && STAGE=2
run $PY -u scripts/train.py --config configs/ucf_ft.yaml --set train.init_from=runs/${MAIN}_s0/best.pt train.stage=$STAGE \
    train.epochs=3 train.warmup_epochs=0.25 model.backbone=vjepa2_l "data.size=[256,256]" > runs/ucf_ft_s0.stdout.log 2>&1
run $PY -u scripts/tune_sigma.py --run runs/ucf_ft_s0 --dataset ucf --split holdout --frame-step 1 --out $OUT/ucf_sigma.json > $OUT/ucf_sigma.log 2>&1
SU=$($PY -c "import json;print(json.load(open('$OUT/ucf_sigma.json'))['best_sigma'])")
run $PY -u scripts/evaluate.py --run runs/ucf_ft_s0 --dataset ucf --split val --metrics all --sigma $SU --out $OUT/ucf_test.json > $OUT/ucf_test.log 2>&1
run $PY -u scripts/evaluate.py --run runs/${MAIN}_s0 --dataset ucf --split val --metrics all --sigma $S1 --out $OUT/ucf_test_zeroshot.json > $OUT/ucf_test_zeroshot.log 2>&1
echo "$(date -Is) final_eval finished"
