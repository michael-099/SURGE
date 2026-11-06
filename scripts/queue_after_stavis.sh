#!/usr/bin/env bash
# After the STAViS queue (one GPU job at a time):
#  1. DHF1K submission format check: predict val videos 601-605 with the submission script, score the PNGs, compare
#     with the ensemble's val per-video CC (must agree within 0.002; research/log.md section 8)
#  2. DHF1K test maps (701-1000) for the submission
#  3. Hollywood-2 (Vig gaze): zero-shot, fine-tune, sigma, test (section 9)
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
ENS="--run runs/main_b1_s0 --run runs/main_b1_s1 --run runs/main_b1_s2"
SUB=D:/DHF1K_submission
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
until [ -f runs/stavis/ft_summe_f3.json ]; do stop; sleep 60; done
until [ -f D:/Datasets/Hollywood2/surge/points.csv ]; do stop; sleep 60; done
echo "$(date -Is) start"

# 1. format check
mkdir -p runs/dhf1k_submission
if [ ! -f runs/dhf1k_submission/format_check.ok ]; then
  run $PY -u scripts/predict_dhf1k_test.py $ENS --sigma 8 --first 601 --last 605 --out $SUB/format_check
  run $PY -u scripts/evaluate.py --maps $SUB/format_check --split val --max-videos 5 --metrics cc,nss \
      --out runs/dhf1k_submission/format_check.json
  run $PY - <<'PY'
import json
a = json.load(open("runs/dhf1k_submission/format_check.json"))["per_video"]
b = json.load(open("runs/final/val_ens3.json"))["per_video"]
d = {k: abs(a[k]["cc"] - b[k]["cc"]) for k in a}
print("per-video |dCC|:", {k: round(v, 5) for k, v in d.items()})
assert len(d) == 5 and max(d.values()) <= 0.002, "format check failed"
open("runs/dhf1k_submission/format_check.ok", "w").write(json.dumps(d))
PY
fi
# 2. test maps
run $PY -u scripts/predict_dhf1k_test.py $ENS --sigma 8 --out $SUB/SURGE-D > runs/dhf1k_submission/predict.log 2>&1
echo "$(date -Is) DHF1K test maps done"

# 3. Hollywood-2 (Vig gaze)
OUT=runs/hollywood_vig; mkdir -p $OUT
[ -f $OUT/zs.json ] || run $PY -u scripts/evaluate.py --run runs/main_b1_s0 --dataset hollywood_vig --split val --metrics all \
    --sigma 12 --out $OUT/zs.json > $OUT/zs.log 2>&1
R=runs/hollywood_vig_ft
if [ ! -f $R/DONE ]; then
  NTR=$($PY -c "import sys;sys.path.insert(0,'.');from surge.data import hollywood_vig_videos as h;print(len(h('train','configs/splits')))")
  CPV=$(( (1500 + NTR - 1) / NTR ))
  mkdir -p $R
  run $PY -u scripts/train.py --config configs/ucf_ft.yaml --set run_dir=$R data.dataset=hollywood_vig \
      train.init_from=runs/main_b1_s0/best.pt train.stage=1 train.epochs=3 train.warmup_epochs=0.25 \
      train.clips_per_video=$CPV eval.frame_step=5 model.backbone=vjepa2_l "data.size=[256,256]" > $R/stdout.log 2>&1
  [ -f $R/DONE ] || { stop; echo "$(date -Is) no DONE in $R"; exit 3; }
fi
[ -f $OUT/ft_sigma.json ] || run $PY -u scripts/tune_sigma.py --run $R --dataset hollywood_vig --split holdout --frame-step 5 \
    --out $OUT/ft_sigma.json > $OUT/ft_sigma.log 2>&1
S=$($PY -c "import json;print(json.load(open('$OUT/ft_sigma.json'))['best_sigma'])")
[ -f $OUT/ft.json ] || run $PY -u scripts/evaluate.py --run $R --dataset hollywood_vig --split val --metrics all --sigma $S \
    --out $OUT/ft.json > $OUT/ft.log 2>&1
echo "$(date -Is) queue_after_stavis finished"
