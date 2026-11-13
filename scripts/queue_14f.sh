#!/usr/bin/env bash
# Section 14F (research/log.md): sound-source localisation maps (EZ-VSL) on the STAViS datasets, ViT-g seed 0.
# Waits for queue 14E to end (one GPU job at a time). Stage 1: Coutrot2, 3 folds; stage 2 (the other 13 folds) only if
# stage 1 passes (scripts/summarize_ssl.py). Per fold: fine-tune with the 11E recipe of runs/g_stavis_<fold>_s0, sigma
# on the holdout, test scored once with matched and once with rotated maps. Stops if SAFETY_STOP.txt (repo root)
# exists. Resumable. Usage: bash scripts/queue_14f.sh > runs/queue_14f.log 2>&1
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
stop() { if [ -f SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; fi; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
sig() { $PY -c "import json;print(json.load(open('$1'))['best_sigma'])"; }
n=0
until grep -q "queue 14E done\|SAFETY_STOP present\|queue stops\|smoke test failed\|not starting" runs/queue_14e.log 2>/dev/null; do
  n=$((n+1)); if [ $n -gt 1500 ]; then echo "$(date -Is) queue 14E not finished after 25 h; not starting"; exit 1; fi
  sleep 60
done
echo "$(date -Is) queue 14E ended: $(tail -1 runs/queue_14e.log)"
stop
G="model.backbone=vjepa2_g data.size=[256,256] train.stage=1 train.epochs=3 train.warmup_epochs=0.25 model.ssl=true"
O=runs/ssl; mkdir -p $O
fold() {
  local F=$1 R=runs/ssl_g_$1
  local NTR CPV FS S
  NTR=$($PY -c "import sys;sys.path.insert(0,'.');from surge.data import stavis_split_videos as s;print(len(s('$F','train','configs/splits')))")
  CPV=$(( (1500 + NTR - 1) / NTR ))
  FS=2; [ "$F" = diem_f1 ] && FS=5
  if [ ! -f $R/DONE ]; then
    mkdir -p $R
    run $PY -u scripts/train.py --config configs/ucf_ft.yaml --set run_dir=$R data.dataset=$F data.root=D:/Datasets \
        train.init_from=runs/vitg_s0/best.pt train.seed=0 train.clips_per_video=$CPV eval.frame_step=$FS $G > $R/stdout.log 2>&1
    [ -f $R/DONE ] || { stop; echo "$(date -Is) no DONE in $R; queue stops"; exit 3; }
  fi
  [ -f $O/sigma_$F.json ] || run $PY -u scripts/tune_sigma.py --run $R --dataset $F --split holdout --frame-step $FS \
      --out $O/sigma_$F.json > $O/sigma_$F.log 2>&1
  S=$(sig $O/sigma_$F.json)
  [ -f $O/stavis_$F.json ] || run $PY -u scripts/evaluate.py --run $R --dataset $F --split val --metrics all --sigma $S \
      --out $O/stavis_$F.json > $O/stavis_$F.log 2>&1
  [ -f $O/stavis_rot_$F.json ] || run $PY -u scripts/evaluate.py --run $R --dataset $F --split val --metrics all --sigma $S \
      --ssl-rotate --out $O/stavis_rot_$F.json > $O/stavis_rot_$F.log 2>&1
}
[ -f D:/Datasets/Coutrot2/surge_ssl/index.csv ] || { echo "$(date -Is) Coutrot2 maps missing; queue stops"; exit 1; }
if [ ! -f runs/smoke_ssl/rot.json ]; then  # GPU smoke test: a few steps, holdout scoring, one rotated-map evaluation
  rm -rf runs/smoke_ssl
  run $PY -u scripts/train.py --config configs/ucf_ft.yaml --set run_dir=runs/smoke_ssl data.dataset=coutrot2_f1 \
      data.root=D:/Datasets train.init_from=runs/vitg_s0/best.pt $G train.epochs=1 train.clips_per_video=2 train.batch=4 \
      train.log_every=2 eval.frame_step=10 > runs/smoke_ssl.log 2>&1
  [ -f runs/smoke_ssl/DONE ] || { echo "$(date -Is) smoke test failed (see runs/smoke_ssl.log)"; exit 1; }
  run $PY -u scripts/evaluate.py --run runs/smoke_ssl --dataset coutrot2_f1 --split val --metrics cc --frame-step 25 \
      --max-videos 2 --ssl-rotate --out runs/smoke_ssl/rot.json >> runs/smoke_ssl.log 2>&1
fi
echo "$(date -Is) smoke test passed"
for F in coutrot2_f1 coutrot2_f2 coutrot2_f3; do fold $F; done
$PY scripts/summarize_ssl.py --stage 1 | tee $O/summary_stage1.txt
grep -q "STAGE1 PASS" $O/summary_stage1.txt || { echo "$(date -Is) stage 1 did not pass (rule 14F); queue 14F done"; exit 0; }
echo "$(date -Is) stage 1 passed: the other five datasets"
for DB in DIEM Coutrot1 ETMD AVAD SumMe; do
  n=0
  until [ -f D:/Datasets/$DB/surge_ssl/index.csv ]; do
    stop; n=$((n+1)); if [ $n -gt 600 ]; then echo "$(date -Is) $DB maps missing after 10 h; queue stops"; exit 1; fi
    sleep 60
  done
done
for F in diem_f1 coutrot1_f1 coutrot1_f2 coutrot1_f3 etmd_f1 etmd_f2 etmd_f3 avad_f1 avad_f2 avad_f3 summe_f1 summe_f2 summe_f3; do
  fold $F
done
$PY scripts/summarize_ssl.py --stage 2 | tee $O/summary_stage2.txt
echo "$(date -Is) queue 14F done"
