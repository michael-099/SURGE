#!/usr/bin/env bash
# research/log.md section 12D (causal SURGE-D, ViT-g), then 12E (component ablations, seeds 1-2), then 12D ViT-L.
# One GPU job at a time. Resumable. Stops if SAFETY_STOP.txt exists or a step fails.
# Usage: bash scripts/queue_review_b.sh > runs/review/queue_b.log 2>&1
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
sig() { $PY -c "import json;print(json.load(open('$1'))['best_sigma'])"; }
cmp() { [ -f $3 ] || $PY scripts/compare.py $1 $2 --out ${3%.txt}.json > $3; }
train() {  # $1 = run dir, rest = train.py args; one retry at batch 4 after out-of-memory
  local R=$1; shift
  [ -f $R/DONE ] && return 0
  stop; mkdir -p $R; echo "$(date -Is) train $R"
  $PY -u scripts/train.py "$@" > $R/stdout.log 2>&1
  if [ ! -f $R/DONE ] && grep -q "OutOfMemory\|out of memory" $R/stdout.log; then
    echo "$(date -Is) $R out of memory; retrying at train.batch=4"
    mv $R/stdout.log $R/stdout_oom.log; rm -f $R/last.pt $R/log.jsonl
    $PY -u scripts/train.py "$@" train.batch=4 > $R/stdout.log 2>&1
  fi
  [ -f $R/DONE ] || { stop; echo "$(date -Is) no DONE in $R; queue stops"; exit 3; }
}
O=runs/review; mkdir -p $O $O/holdout
REC="train.epochs=3 train.warmup_epochs=0.25 data.size=[256,256]"
SALMAPS=D:/Research/baselines/salfom_val/maps

# SalFoM's best blur from 12C (CPU job started earlier); wait for it up to 3 hours, else use sigma 0 and say so.
salfom_best() {
  local t=0
  until [ -f $O/salfom_val_sigma24.json ] || [ $t -ge 180 ]; do sleep 60; t=$((t + 1)); done
  $PY -c "
import json, os
r = {s: json.load(open(f'$O/salfom_val_sigma{s}.json'))['mean']['cc'] for s in (0, 4, 8, 12, 16, 24)
     if os.path.exists(f'$O/salfom_val_sigma{s}.json')}
print(max(r, key=lambda s: (round(r[s], 4), -s)) if len(r) == 6 else 0)"
}

causal_block() {  # $1 = tag (g or l), $2 = backbone
  local tag=$1 bb=$2
  for s in 0 1 2; do
    train runs/causal_${tag}_s$s --config configs/main_b1.yaml --set run_dir=runs/causal_${tag}_s$s train.seed=$s \
        model.backbone=$bb data.target=last eval.frame_step=10 $REC
  done
  local ENS="--run runs/causal_${tag}_s0 --run runs/causal_${tag}_s1 --run runs/causal_${tag}_s2"
  [ -f $O/causal_${tag}_sigma_s0.json ] || run $PY -u scripts/tune_sigma.py --run runs/causal_${tag}_s0 --frame-step 10 \
      --out $O/causal_${tag}_sigma_s0.json > $O/causal_${tag}_sigma_s0.log 2>&1
  [ -f $O/causal_${tag}_sigma_ens3.json ] || run $PY -u scripts/tune_sigma.py $ENS --frame-step 10 \
      --out $O/causal_${tag}_sigma_ens3.json > $O/causal_${tag}_sigma_ens3.log 2>&1
  local S1 SE; S1=$(sig $O/causal_${tag}_sigma_s0.json); SE=$(sig $O/causal_${tag}_sigma_ens3.json)
  for s in 0 1 2; do
    [ -f $O/causal_${tag}_val_s$s.json ] || run $PY -u scripts/evaluate.py --run runs/causal_${tag}_s$s --split val \
        --metrics all --sigma $S1 --frame-step 10 --out $O/causal_${tag}_val_s$s.json > $O/causal_${tag}_val_s$s.log 2>&1
  done
  [ -f $O/causal_${tag}_val_ens3.json ] || run $PY -u scripts/evaluate.py $ENS --split val --metrics all --sigma $SE \
      --frame-step 10 --out $O/causal_${tag}_val_ens3.json > $O/causal_${tag}_val_ens3.log 2>&1
}
causal_compare() {  # $1 = tag; needs SalFoM on the same frames
  local tag=$1
  for x in s0 s1 s2 ens3; do
    cmp $O/causal_${tag}_val_$x.json $O/salfom_val_step10_best.json $O/cmp_causal_${tag}_${x}_vs_salfom_best.txt
    cmp $O/causal_${tag}_val_$x.json $O/salfom_val_step10_sigma0.json $O/cmp_causal_${tag}_${x}_vs_salfom.txt
  done
}

# ------------------------------------------------------------------ 12D causal ViT-g (primary test of RQ1)
causal_block g vjepa2_g
echo "$(date -Is) 12D ViT-g scored"

# ------------------------------------------------------------------ SalFoM on the same every-10th val frames (CPU)
SB=$(salfom_best); echo "$(date -Is) SalFoM best sigma (12C): $SB"; echo $SB > $O/salfom_best_sigma.txt
[ -f $O/salfom_val_step10_sigma0.json ] || run $PY -u scripts/evaluate.py --maps $SALMAPS --split val --metrics all \
    --frame-step 10 --out $O/salfom_val_step10_sigma0.json > $O/salfom_val_step10_sigma0.log 2>&1
[ -f $O/salfom_val_step10_best.json ] || run $PY -u scripts/evaluate.py --maps $SALMAPS --split val --metrics all \
    --frame-step 10 --sigma $SB --out $O/salfom_val_step10_best.json > $O/salfom_val_step10_best.log 2>&1
causal_compare g
echo "$(date -Is) 12D ViT-g compared"

# offline ViT-g on the same frames, for causal vs offline
[ -f $O/offline_g_val_step10_s0.json ] || run $PY -u scripts/evaluate.py --run runs/vitg_s0 --split val --metrics all \
    --sigma 8 --frame-step 10 --out $O/offline_g_val_step10_s0.json > $O/offline_g_val_step10_s0.log 2>&1
[ -f $O/offline_g_val_step10_ens3.json ] || run $PY -u scripts/evaluate.py --run runs/vitg_s0 --run runs/vitg_s1 \
    --run runs/vitg_s2 --split val --metrics all --sigma 8 --frame-step 10 --out $O/offline_g_val_step10_ens3.json \
    > $O/offline_g_val_step10_ens3.log 2>&1
cmp $O/causal_g_val_s0.json $O/offline_g_val_step10_s0.json $O/cmp_causal_vs_offline_g_s0.txt
cmp $O/causal_g_val_ens3.json $O/offline_g_val_step10_ens3.json $O/cmp_causal_vs_offline_g_ens3.txt
cmp $O/offline_g_val_step10_ens3.json $O/salfom_val_step10_best.json $O/cmp_offline_g_ens3_step10_vs_salfom_best.txt

# ------------------------------------------------------------------ 12E component ablations, seeds 1 and 2 (ViT-L)
for f in noprior:model.prior=false noaux:model.aux=false salicon:data.salicon=true stride4:data.stride=4; do
  name=${f%%:*}; ov=${f#*:}
  for s in 1 2; do
    train runs/abl_${name}_s$s --config configs/main_b1.yaml --set run_dir=runs/abl_${name}_s$s train.seed=$s \
        model.backbone=vjepa2_l $ov $REC
  done
done
for R in main_b1_s0 main_b1_s1 main_b1_s2 abl_noprior abl_noaux abl_salicon abl_stride4 \
         abl_noprior_s1 abl_noprior_s2 abl_noaux_s1 abl_noaux_s2 abl_salicon_s1 abl_salicon_s2 abl_stride4_s1 abl_stride4_s2; do
  [ -f $O/holdout/$R.json ] || run $PY -u scripts/evaluate.py --run runs/$R --split holdout --metrics cc,nss,sim,kld \
      --frame-step 5 --sigma 0 --out $O/holdout/$R.json > $O/holdout/$R.log 2>&1
done
echo "$(date -Is) 12E done"

# ------------------------------------------------------------------ 12D causal ViT-L
causal_block l vjepa2_l
causal_compare l
echo "$(date -Is) queue B done"
