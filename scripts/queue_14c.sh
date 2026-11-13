#!/usr/bin/env bash
# Section 14C (research/log.md): the adopted model, SURGE-D ViT-g + expert $EXP, evaluated exactly like SURGE-D ViT-g
# (sections 10B, 11E, 12A): DHF1K val (each seed and the three-seed ensemble), UCF Sports, Hollywood-2 (Vig gaze) and
# the 16 STAViS folds, three seeds each, output blur tuned on each holdout; paired comparisons with SURGE-D ViT-g and
# SalFoM. Launched only after the three-seed decision of 14A. Resumable: finished outputs are skipped.
#   EXP=dinov2_l bash scripts/queue_14c.sh > runs/queue_14c.log 2>&1
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
EXP=${EXP:?set EXP=dinov2_l or umt_l}
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; return 0; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
sig() { $PY -c "import json;print(json.load(open('$1'))['best_sigma'])"; }
train() {  # $1 = run dir, rest = train.py args; retries once at batch 4 after running out of memory
  local R=$1; shift
  [ -f $R/DONE ] && return 0
  stop; mkdir -p $R; echo "$(date -Is) train $R"
  $PY -u scripts/train.py "$@" > $R/stdout.log 2>&1
  if [ ! -f $R/DONE ] && grep -q "OutOfMemory\|out of memory" $R/stdout.log; then
    echo "$(date -Is) $R out of memory at the default batch; retrying at train.batch=4"
    mv $R/stdout.log $R/stdout_oom.log
    $PY -u scripts/train.py "$@" train.batch=4 > $R/stdout.log 2>&1
  fi
  [ -f $R/DONE ] || { stop; echo "$(date -Is) no DONE in $R; queue stops"; exit 3; }
}
cmp() { [ -f $3 ] || $PY scripts/compare.py $1 $2 --out ${3%.txt}.json > $3; }
SZ="data.size=[256,256]"
G="model.backbone=vjepa2_g $SZ model.expert=$EXP train.stage=1 train.epochs=3 train.warmup_epochs=0.25"
X=runs/xp_${EXP}
O=runs/xp_final_${EXP}; mkdir -p $O
for s in 0 1 2; do [ -f ${X}_s$s/DONE ] || { echo "$(date -Is) missing ${X}_s$s"; exit 3; }; done

# ------------------------------------------------------------------ DHF1K val
E3="--run ${X}_s0 --run ${X}_s1 --run ${X}_s2"
[ -f $O/sigma_single.json ] || run $PY -u scripts/tune_sigma.py --run ${X}_s0 --out $O/sigma_single.json > $O/sigma_single.log 2>&1
[ -f $O/sigma_ens3.json ] || run $PY -u scripts/tune_sigma.py $E3 --out $O/sigma_ens3.json > $O/sigma_ens3.log 2>&1
for s in 0 1 2; do
  [ -f $O/val_single_s$s.json ] || run $PY -u scripts/evaluate.py --run ${X}_s$s --split val --metrics all \
      --sigma $(sig $O/sigma_single.json) --out $O/val_single_s$s.json > $O/val_single_s$s.log 2>&1
done
[ -f $O/val_ens3.json ] || run $PY -u scripts/evaluate.py $E3 --split val --metrics all --sigma $(sig $O/sigma_ens3.json) \
    --out $O/val_ens3.json > $O/val_ens3.log 2>&1
cmp $O/val_single_s0.json runs/metric_check/salfom_val_all.json $O/cmp_single_vs_salfom.txt
cmp $O/val_ens3.json runs/metric_check/salfom_val_all.json $O/cmp_ens3_vs_salfom.txt
cmp $O/val_single_s0.json runs/review/salfom_val_best_all.json $O/cmp_single_vs_salfom_best.txt
cmp $O/val_ens3.json runs/review/salfom_val_best_all.json $O/cmp_ens3_vs_salfom_best.txt
cmp $O/val_single_s0.json runs/vitg/val_single_s0.json $O/cmp_single_vs_surged.txt
cmp $O/val_ens3.json runs/vitg/val_ens3.json $O/cmp_ens3_vs_surged.txt
echo "$(date -Is) 14C DHF1K val done"

# ------------------------------------------------------------------ UCF Sports
for k in 0 1 2; do
  train ${X}_ucf_s$k --config configs/ucf_ft.yaml --set run_dir=${X}_ucf_s$k train.seed=$k train.init_from=${X}_s$k/best.pt $G
done
U="--run ${X}_ucf_s0 --run ${X}_ucf_s1 --run ${X}_ucf_s2"
[ -f $O/ucf_sigma_s0.json ] || run $PY -u scripts/tune_sigma.py --run ${X}_ucf_s0 --dataset ucf --split holdout --frame-step 1 --out $O/ucf_sigma_s0.json > $O/ucf_sigma_s0.log 2>&1
[ -f $O/ucf_sigma_ens.json ] || run $PY -u scripts/tune_sigma.py $U --dataset ucf --split holdout --frame-step 1 --out $O/ucf_sigma_ens.json > $O/ucf_sigma_ens.log 2>&1
[ -f $O/ucf_s0.json ] || run $PY -u scripts/evaluate.py --run ${X}_ucf_s0 --dataset ucf --split val --metrics all --sigma $(sig $O/ucf_sigma_s0.json) --out $O/ucf_s0.json > $O/ucf_s0.log 2>&1
[ -f $O/ucf_ens.json ] || run $PY -u scripts/evaluate.py $U --dataset ucf --split val --metrics all --sigma $(sig $O/ucf_sigma_ens.json) --out $O/ucf_ens.json > $O/ucf_ens.log 2>&1
cmp $O/ucf_s0.json runs/g_other/ucf_s0.json $O/cmp_ucf_s0_vs_surged.txt
cmp $O/ucf_ens.json runs/g_other/ucf_ens.json $O/cmp_ucf_ens_vs_surged.txt
echo "$(date -Is) 14C UCF done"

# ------------------------------------------------------------------ Hollywood-2 (Vig gaze)
for k in 0 1 2; do
  train ${X}_hollywood_s$k --config configs/ucf_ft.yaml --set run_dir=${X}_hollywood_s$k train.seed=$k \
      data.dataset=hollywood_vig train.init_from=${X}_s$k/best.pt train.clips_per_video=3 eval.frame_step=5 $G
done
H="--run ${X}_hollywood_s0 --run ${X}_hollywood_s1 --run ${X}_hollywood_s2"
[ -f $O/hollywood_sigma_s0.json ] || run $PY -u scripts/tune_sigma.py --run ${X}_hollywood_s0 --dataset hollywood_vig --split holdout --frame-step 5 --out $O/hollywood_sigma_s0.json > $O/hollywood_sigma_s0.log 2>&1
[ -f $O/hollywood_sigma_ens.json ] || run $PY -u scripts/tune_sigma.py $H --dataset hollywood_vig --split holdout --frame-step 5 --out $O/hollywood_sigma_ens.json > $O/hollywood_sigma_ens.log 2>&1
[ -f $O/hollywood_s0.json ] || run $PY -u scripts/evaluate.py --run ${X}_hollywood_s0 --dataset hollywood_vig --split val --metrics all --sigma $(sig $O/hollywood_sigma_s0.json) --out $O/hollywood_s0.json > $O/hollywood_s0.log 2>&1
[ -f $O/hollywood_ens.json ] || run $PY -u scripts/evaluate.py $H --dataset hollywood_vig --split val --metrics all --sigma $(sig $O/hollywood_sigma_ens.json) --out $O/hollywood_ens.json > $O/hollywood_ens.log 2>&1
cmp $O/hollywood_s0.json runs/g_other/hollywood_s0.json $O/cmp_hollywood_s0_vs_surged.txt
cmp $O/hollywood_ens.json runs/g_other/hollywood_ens.json $O/cmp_hollywood_ens_vs_surged.txt
echo "$(date -Is) 14C Hollywood done"

# ------------------------------------------------------------------ the 16 STAViS folds
FOLDS="coutrot2_f1 coutrot2_f2 coutrot2_f3 etmd_f1 etmd_f2 etmd_f3 avad_f1 avad_f2 avad_f3 coutrot1_f1 coutrot1_f2 coutrot1_f3 summe_f1 summe_f2 summe_f3 diem_f1"
for F in $FOLDS; do
  NTR=$($PY -c "import sys;sys.path.insert(0,'.');from surge.data import stavis_split_videos as s;print(len(s('$F','train','configs/splits')))")
  CPV=$(( (1500 + NTR - 1) / NTR ))
  FS=2; [ "$F" = diem_f1 ] && FS=5
  for k in 0 1 2; do
    train ${X}_stavis_${F}_s$k --config configs/ucf_ft.yaml --set run_dir=${X}_stavis_${F}_s$k train.seed=$k \
        data.dataset=$F data.root=D:/Datasets train.init_from=${X}_s$k/best.pt train.clips_per_video=$CPV eval.frame_step=$FS $G
  done
  M="--run ${X}_stavis_${F}_s0 --run ${X}_stavis_${F}_s1 --run ${X}_stavis_${F}_s2"
  [ -f $O/stavis_sigma_s0_$F.json ] || run $PY -u scripts/tune_sigma.py --run ${X}_stavis_${F}_s0 --dataset $F --split holdout --frame-step $FS --out $O/stavis_sigma_s0_$F.json > $O/stavis_sigma_s0_$F.log 2>&1
  [ -f $O/stavis_sigma_ens_$F.json ] || run $PY -u scripts/tune_sigma.py $M --dataset $F --split holdout --frame-step $FS --out $O/stavis_sigma_ens_$F.json > $O/stavis_sigma_ens_$F.log 2>&1
  [ -f $O/stavis_s0_$F.json ] || run $PY -u scripts/evaluate.py --run ${X}_stavis_${F}_s0 --dataset $F --split val --metrics all --sigma $(sig $O/stavis_sigma_s0_$F.json) --out $O/stavis_s0_$F.json > $O/stavis_s0_$F.log 2>&1
  [ -f $O/stavis_ens_$F.json ] || run $PY -u scripts/evaluate.py $M --dataset $F --split val --metrics all --sigma $(sig $O/stavis_sigma_ens_$F.json) --out $O/stavis_ens_$F.json > $O/stavis_ens_$F.log 2>&1
done
echo "$(date -Is) 14C STAViS done"
echo "$(date -Is) queue 14C finished"
