#!/usr/bin/env bash
# STAViS datasets (research/log.md section 7), one GPU job at a time. Resumable: finished outputs are skipped.
#  A. zero-shot: main DHF1K model (B1 seed 0, sigma 12) on every test fold
#  B. per fold: fine-tune from B1 seed 0, tune sigma on the fold's holdout, score the fold's test list once
# Stops if SAFETY_STOP.txt exists or a step fails. Usage: bash scripts/queue_stavis.sh > runs/queue_stavis.log 2>&1
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
OUT=runs/stavis; mkdir -p $OUT
FOLDS="diem_f1 coutrot1_f1 coutrot1_f2 coutrot1_f3 coutrot2_f1 coutrot2_f2 coutrot2_f3 etmd_f1 etmd_f2 etmd_f3 avad_f1 avad_f2 avad_f3 summe_f1 summe_f2 summe_f3"
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }

for F in $FOLDS; do
  [ -f $OUT/zs_$F.json ] && continue
  run $PY -u scripts/evaluate.py --run runs/main_b1_s0 --dataset $F --split val --metrics all --sigma 12 \
      --out $OUT/zs_$F.json > $OUT/zs_$F.log 2>&1
done
echo "$(date -Is) zero-shot done"

for F in $FOLDS; do
  R=runs/stavis_ft_$F
  if [ ! -f $R/DONE ]; then
    NTR=$($PY -c "import sys;sys.path.insert(0,'.');from surge.data import stavis_split_videos as s;print(len(s('$F','train','configs/splits')))")
    CPV=$(( (1500 + NTR - 1) / NTR ))
    FS=2; [ "$F" = diem_f1 ] && FS=5
    mkdir -p $R
    run $PY -u scripts/train.py --config configs/ucf_ft.yaml --set run_dir=$R data.dataset=$F data.root=D:/Datasets \
        train.init_from=runs/main_b1_s0/best.pt train.stage=1 train.epochs=3 train.warmup_epochs=0.25 \
        train.clips_per_video=$CPV eval.frame_step=$FS model.backbone=vjepa2_l "data.size=[256,256]" > $R/stdout.log 2>&1
    [ -f $R/DONE ] || { stop; echo "$(date -Is) no DONE in $R; queue stops"; exit 3; }
  fi
  FS=2; [ "$F" = diem_f1 ] && FS=5
  [ -f $OUT/ft_sigma_$F.json ] || run $PY -u scripts/tune_sigma.py --run $R --dataset $F --split holdout --frame-step $FS \
      --out $OUT/ft_sigma_$F.json > $OUT/ft_sigma_$F.log 2>&1
  S=$($PY -c "import json;print(json.load(open('$OUT/ft_sigma_$F.json'))['best_sigma'])")
  [ -f $OUT/ft_$F.json ] || run $PY -u scripts/evaluate.py --run $R --dataset $F --split val --metrics all --sigma $S \
      --out $OUT/ft_$F.json > $OUT/ft_$F.log 2>&1
done
echo "$(date -Is) queue_stavis finished"
