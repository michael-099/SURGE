#!/usr/bin/env bash
# research/log.md section 13A: learning curve over the number of training videos (ViT-L, 135 and 270 videos, seeds 0-2;
# the 540-video arm is runs/main_b1_s{0,1,2}), then the holdout rescoring and the analysis.
# One GPU job at a time. Resumable. Stops if SAFETY_STOP.txt exists or a step fails.
# Usage: bash scripts/queue_review_d.sh > runs/review/queue_d.log 2>&1
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
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
REC="model.backbone=vjepa2_l train.epochs=3 train.warmup_epochs=0.25 data.size=[256,256]"

for s in 0 1 2; do
  for n in 270 135; do
    [ -f configs/splits/dhf1k_train${n}_s$s.txt ] || { echo "$(date -Is) missing split $n s$s; queue stops"; exit 3; }
    cpv=$((540 * 8 / n))
    train runs/lc${n}_s$s --config configs/main_b1.yaml --set run_dir=runs/lc${n}_s$s train.seed=$s \
        data.train_split=configs/splits/dhf1k_train${n}_s$s.txt train.clips_per_video=$cpv $REC
  done
done

# holdout per-video scores (every 5th frame, sigma 0, as during training and as in 12E)
for s in 0 1 2; do
  for n in 270 135; do
    R=lc${n}_s$s
    [ -f $O/holdout/$R.json ] || run $PY -u scripts/evaluate.py --run runs/$R --split holdout --metrics cc,nss,sim,kld \
        --frame-step 5 --sigma 0 --out $O/holdout/$R.json > $O/holdout/$R.log 2>&1
    [ -f $O/holdout/$R.json ] || { echo "$(date -Is) no $O/holdout/$R.json; queue stops"; exit 3; }
  done
done
[ -f scripts/learning_curve.py ] && run $PY scripts/learning_curve.py --out $O/learning_curve.json > $O/learning_curve.txt 2>&1
echo "$(date -Is) queue D done"
