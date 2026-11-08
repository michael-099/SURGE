#!/usr/bin/env bash
# research/log.md section 11D: audio-conditioned fine-tunes on the 16 STAViS folds (ViT-L, seed 0, section 7 recipe),
# sigma on the holdout, each test fold scored once with matched audio and once with rotated audio (control).
# Called by queue_next.sh (one GPU job at a time). Resumable.
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
sig() { $PY -c "import json;print(json.load(open('$1'))['best_sigma'])"; }
for DB in DIEM Coutrot1 Coutrot2 ETMD AVAD SumMe; do
  until [ -f D:/Datasets/$DB/surge_audio/index.csv ]; do stop; sleep 60; done
done
O=runs/av; mkdir -p $O
FOLDS="diem_f1 coutrot1_f1 coutrot1_f2 coutrot1_f3 coutrot2_f1 coutrot2_f2 coutrot2_f3 etmd_f1 etmd_f2 etmd_f3 avad_f1 avad_f2 avad_f3 summe_f1 summe_f2 summe_f3"
for F in $FOLDS; do
  R=runs/stavis_av_$F
  NTR=$($PY -c "import sys;sys.path.insert(0,'.');from surge.data import stavis_split_videos as s;print(len(s('$F','train','configs/splits')))")
  CPV=$(( (1500 + NTR - 1) / NTR ))
  FS=2; [ "$F" = diem_f1 ] && FS=5
  if [ ! -f $R/DONE ]; then
    mkdir -p $R
    run $PY -u scripts/train.py --config configs/ucf_ft.yaml --set run_dir=$R data.dataset=$F data.root=D:/Datasets \
        train.init_from=runs/main_b1_s0/best.pt train.stage=1 train.epochs=3 train.warmup_epochs=0.25 \
        train.clips_per_video=$CPV eval.frame_step=$FS model.backbone=vjepa2_l "data.size=[256,256]" model.audio=true > $R/stdout.log 2>&1
    [ -f $R/DONE ] || { stop; echo "$(date -Is) no DONE in $R; queue stops"; exit 3; }
  fi
  [ -f $O/sigma_$F.json ] || run $PY -u scripts/tune_sigma.py --run $R --dataset $F --split holdout --frame-step $FS \
      --out $O/sigma_$F.json > $O/sigma_$F.log 2>&1
  S=$(sig $O/sigma_$F.json)
  [ -f $O/stavis_$F.json ] || run $PY -u scripts/evaluate.py --run $R --dataset $F --split val --metrics all --sigma $S \
      --out $O/stavis_$F.json > $O/stavis_$F.log 2>&1
  [ -f $O/stavis_rot_$F.json ] || run $PY -u scripts/evaluate.py --run $R --dataset $F --split val --metrics all --sigma $S \
      --audio-rotate --out $O/stavis_rot_$F.json > $O/stavis_rot_$F.log 2>&1
done
echo "$(date -Is) queue_audio finished"
