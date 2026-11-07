#!/usr/bin/env bash
# research/log.md section 10, one GPU job at a time. Resumable: finished outputs are skipped.
#  10B  V-JEPA 2 ViT-g: G-1 on DHF1K (B1 recipe, seed 0) -> holdout gate (>= 0.5833 CC); if passed: seeds 1-2,
#       sigma, DHF1K val single + ensemble, UCF fine-tune + test.
#  10A  ViT-L seed ensembles (members 1, 2 from main_b1_s1/s2) on UCF, the 16 STAViS folds and Hollywood-2 (Vig gaze).
# Stops if SAFETY_STOP.txt exists or a step fails. Usage: bash scripts/queue_improve.sh > runs/queue_improve.log 2>&1
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
sig() { $PY -c "import json;print(json.load(open('$1'))['best_sigma'])"; }
SZ="data.size=[256,256]"

# trains one run; $1 = run dir, rest = train.py args. Retries once at batch 4 if the first attempt ran out of memory.
train() {
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

# ------------------------------------------------------------------------------------------------ 10B ViT-g
O=runs/vitg; mkdir -p $O
until $PY -c "import sys;from huggingface_hub import snapshot_download as s;s('facebook/vjepa2-vitg-fpc64-256',local_files_only=True)" 2>/dev/null; do
  stop; sleep 60; done
for s in 0 1 2; do
  if [ $s -gt 0 ]; then
    G=$($PY -c "import json;d=[json.loads(l) for l in open('runs/vitg_s0/log.jsonl')];print(int([x for x in d if x['event']=='done'][-1]['best']>=0.5833))")
    if [ "$G" != 1 ]; then echo "$(date -Is) ViT-g gate FAILED (G-1 holdout below 0.5833); DHF1K val not touched"; break; fi
    [ $s = 1 ] && echo "$(date -Is) ViT-g gate PASSED"
  fi
  train runs/vitg_s$s --config configs/main_b1.yaml --set run_dir=runs/vitg_s$s train.seed=$s train.epochs=3 \
      train.warmup_epochs=0.25 model.backbone=vjepa2_g "$SZ"
done
if [ -f runs/vitg_s2/DONE ]; then
  [ -f $O/sigma_single.json ] || run $PY -u scripts/tune_sigma.py --run runs/vitg_s0 --out $O/sigma_single.json > $O/sigma_single.log 2>&1
  [ -f $O/sigma_ens3.json ] || run $PY -u scripts/tune_sigma.py --run runs/vitg_s0 --run runs/vitg_s1 --run runs/vitg_s2 \
      --out $O/sigma_ens3.json > $O/sigma_ens3.log 2>&1
  [ -f $O/val_single_s0.json ] || run $PY -u scripts/evaluate.py --run runs/vitg_s0 --split val --metrics all \
      --sigma $(sig $O/sigma_single.json) --out $O/val_single_s0.json > $O/val_single_s0.log 2>&1
  [ -f $O/val_ens3.json ] || run $PY -u scripts/evaluate.py --run runs/vitg_s0 --run runs/vitg_s1 --run runs/vitg_s2 \
      --split val --metrics all --sigma $(sig $O/sigma_ens3.json) --out $O/val_ens3.json > $O/val_ens3.log 2>&1
  $PY scripts/compare.py $O/val_single_s0.json runs/metric_check/salfom_val_all.json --out $O/cmp_single_vs_salfom.json > $O/cmp_single_vs_salfom.txt
  $PY scripts/compare.py $O/val_ens3.json runs/metric_check/salfom_val_all.json --out $O/cmp_ens3_vs_salfom.json > $O/cmp_ens3_vs_salfom.txt
  train runs/ucf_ft_vitg_s0 --config configs/ucf_ft.yaml --set run_dir=runs/ucf_ft_vitg_s0 train.init_from=runs/vitg_s0/best.pt \
      train.stage=1 train.epochs=3 train.warmup_epochs=0.25 model.backbone=vjepa2_g "$SZ"
  [ -f $O/ucf_sigma.json ] || run $PY -u scripts/tune_sigma.py --run runs/ucf_ft_vitg_s0 --dataset ucf --split holdout \
      --frame-step 1 --out $O/ucf_sigma.json > $O/ucf_sigma.log 2>&1
  [ -f $O/ucf_test.json ] || run $PY -u scripts/evaluate.py --run runs/ucf_ft_vitg_s0 --dataset ucf --split val --metrics all \
      --sigma $(sig $O/ucf_sigma.json) --out $O/ucf_test.json > $O/ucf_test.log 2>&1
fi
echo "$(date -Is) 10B done"

# ------------------------------------------------------------------------------------------------ 10A ensembles
E=runs/ens; mkdir -p $E
# UCF Sports
for k in 1 2; do
  train runs/ucf_ft_s$k --config configs/ucf_ft.yaml --set run_dir=runs/ucf_ft_s$k train.seed=$k \
      train.init_from=runs/main_b1_s$k/best.pt train.stage=1 train.epochs=3 train.warmup_epochs=0.25 model.backbone=vjepa2_l "$SZ"
done
U="--run runs/ucf_ft_s0 --run runs/ucf_ft_s1 --run runs/ucf_ft_s2"
[ -f $E/ucf_sigma.json ] || run $PY -u scripts/tune_sigma.py $U --dataset ucf --split holdout --frame-step 1 --out $E/ucf_sigma.json > $E/ucf_sigma.log 2>&1
[ -f $E/ucf_test.json ] || run $PY -u scripts/evaluate.py $U --dataset ucf --split val --metrics all \
    --sigma $(sig $E/ucf_sigma.json) --out $E/ucf_test.json > $E/ucf_test.log 2>&1
echo "$(date -Is) UCF ensemble done"

# STAViS folds
FOLDS="diem_f1 coutrot1_f1 coutrot1_f2 coutrot1_f3 coutrot2_f1 coutrot2_f2 coutrot2_f3 etmd_f1 etmd_f2 etmd_f3 avad_f1 avad_f2 avad_f3 summe_f1 summe_f2 summe_f3"
for F in $FOLDS; do
  NTR=$($PY -c "import sys;sys.path.insert(0,'.');from surge.data import stavis_split_videos as s;print(len(s('$F','train','configs/splits')))")
  CPV=$(( (1500 + NTR - 1) / NTR ))
  FS=2; [ "$F" = diem_f1 ] && FS=5
  for k in 1 2; do
    train runs/stavis_ft_${F}_s$k --config configs/ucf_ft.yaml --set run_dir=runs/stavis_ft_${F}_s$k train.seed=$k \
        data.dataset=$F data.root=D:/Datasets train.init_from=runs/main_b1_s$k/best.pt train.stage=1 train.epochs=3 \
        train.warmup_epochs=0.25 train.clips_per_video=$CPV eval.frame_step=$FS model.backbone=vjepa2_l "$SZ"
  done
  M="--run runs/stavis_ft_$F --run runs/stavis_ft_${F}_s1 --run runs/stavis_ft_${F}_s2"
  [ -f $E/stavis_sigma_$F.json ] || run $PY -u scripts/tune_sigma.py $M --dataset $F --split holdout --frame-step $FS \
      --out $E/stavis_sigma_$F.json > $E/stavis_sigma_$F.log 2>&1
  [ -f $E/stavis_$F.json ] || run $PY -u scripts/evaluate.py $M --dataset $F --split val --metrics all \
      --sigma $(sig $E/stavis_sigma_$F.json) --out $E/stavis_$F.json > $E/stavis_$F.log 2>&1
done
echo "$(date -Is) STAViS ensembles done"

# Hollywood-2 (Vig gaze)
for k in 1 2; do
  train runs/hollywood_vig_ft_s$k --config configs/ucf_ft.yaml --set run_dir=runs/hollywood_vig_ft_s$k train.seed=$k \
      data.dataset=hollywood_vig train.init_from=runs/main_b1_s$k/best.pt train.stage=1 train.epochs=3 \
      train.warmup_epochs=0.25 train.clips_per_video=3 eval.frame_step=5 model.backbone=vjepa2_l "$SZ"
done
H="--run runs/hollywood_vig_ft --run runs/hollywood_vig_ft_s1 --run runs/hollywood_vig_ft_s2"
[ -f $E/hollywood_sigma.json ] || run $PY -u scripts/tune_sigma.py $H --dataset hollywood_vig --split holdout --frame-step 5 \
    --out $E/hollywood_sigma.json > $E/hollywood_sigma.log 2>&1
[ -f $E/hollywood_test.json ] || run $PY -u scripts/evaluate.py $H --dataset hollywood_vig --split val --metrics all \
    --sigma $(sig $E/hollywood_sigma.json) --out $E/hollywood_test.json > $E/hollywood_test.log 2>&1
echo "$(date -Is) queue_improve finished"
