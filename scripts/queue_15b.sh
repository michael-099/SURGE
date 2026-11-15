#!/usr/bin/env bash
# Section 15 (research/log.md), training block: smoke tests of scripts/train_multi.py, then 15B (linear probes),
# 15C/D (decoder ablations and training configuration, seeds 0-2), 15E2 (static model, seeds 0-2), 15F3 (384 pixels),
# 15A2 (reproduction of seed 0), 15I (zero-shot generalisation; maps saved for 15M) and 15A3 (cost). Waits for queue 15t (one GPU job at a
# time). Stops if SAFETY_STOP.txt (repo root) exists. Resumable (finished runs and outputs are skipped).
# Usage: bash scripts/queue_15b.sh > runs/inv/queue_15b.log 2>&1
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
O=runs/review/inv
L=runs/inv
mkdir -p $O $L $O/holdout
stop() { if [ -f SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; fi; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
n=0  # waits for queue 15l (section 15L), which itself runs after queue 15t
until grep -q "queue 15l done\|SAFETY_STOP present\|queue stops\|FAILED\|not starting" $L/queue_15l.log 2>/dev/null; do
  n=$((n+1)); if [ $n -gt 1200 ]; then echo "$(date -Is) queue 15l not finished after 20 h; not starting"; exit 1; fi
  sleep 60
done
grep -q "queue 15l done" $L/queue_15l.log || { echo "$(date -Is) queue 15l did not finish cleanly; not starting"; exit 1; }
sleep 20
echo "$(date -Is) queue 15l done; starting queue 15b"
G="model.backbone=vjepa2_g train.epochs=3 train.warmup_epochs=0.25 data.size=[256,256]"
TM="$PY -u scripts/train_multi.py --config configs/main_b1.yaml --set"
# ---- smoke tests (ViT-L, 1 epoch, 1 clip per video, coarse holdout scoring)
if [ ! -f runs/inv_smoke_dec/DONE ]; then
  rm -rf runs/inv_smoke_dec
  run $TM run_dir=runs/inv_smoke_dec model.backbone=vjepa2_l train.epochs=1 train.clips_per_video=1 eval.frame_step=50 \
      --variants R,V1,V2,V3,V4,V5,V6,nocc,R6 > $L/smoke_dec.log 2>&1
fi
if [ ! -f runs/inv_smoke_probe/DONE ]; then
  rm -rf runs/inv_smoke_probe
  run $TM run_dir=runs/inv_smoke_probe model.backbone=vjepa2_l train.epochs=1 train.clips_per_video=1 eval.frame_step=50 \
      train.lr=1e-3 --probes 1-24 > $L/smoke_probe.log 2>&1
fi
# saved low-resolution maps reproduce the evaluator (section 15M)
if [ ! -f $L/smoke_small/OK ]; then
  rm -rf $L/smoke_small
  run $PY -u scripts/evaluate.py --run runs/vitg_s0 --metrics cc,nss,sim,kld --split holdout --frame-step 25 \
      --max-videos 2 --save-small $L/smoke_small --out $L/smoke_small/zs.json > $L/smoke_small.log 2>&1
  run env CUDA_VISIBLE_DEVICES="" $PY scripts/inv_bias.py --selfcheck $L/smoke_small/zs.json $L/smoke_small dhf1k \
      holdout >> $L/smoke_small.log 2>&1
  touch $L/smoke_small/OK
fi
echo "$(date -Is) smoke tests passed"
# ---- 15I first part: MVVA (its maps feed 15K) and AViMoS, zero-shot with runs/vitg_s0
[ -f $O/zs_mvva.json ] || run $PY -u scripts/probe_eval.py --run runs/vitg_s0 --dataset mvva --split val --sigma 8 \
    --frame-step 5 --save-small runs/inv/mvva_small_s0 --out $O/zs_mvva.json > $L/zs_mvva.log 2>&1
[ -f $O/zs_avimos.json ] || run $PY -u scripts/evaluate.py --run runs/vitg_s0 --metrics cc,nss,sim,kld --dataset avimos \
    --split val --sigma 12 --frame-step 5 --save-small $L/zs_small/avimos --out $O/zs_avimos.json > $L/zs_avimos.log 2>&1
# ---- 15B: linear probes per block (ViT-g, seed 0)
[ -f runs/inv_b_s0/DONE ] || run $TM run_dir=runs/inv_b_s0 train.seed=0 $G train.lr=1e-3 --probes 1-40 > $L/inv_b_s0.log 2>&1
# ---- 15C/D: decoder ablations and training configuration (seeds 0-2; seed 0 also trains R for 6 epochs)
for s in 0 1 2; do
  V=R,V1,V2,V3,V4,V5,V6,lr1e-4,lr9e-4,nocc,nonss
  [ $s = 0 ] && V=$V,R6
  [ -f runs/inv_c_s$s/DONE ] || run $TM run_dir=runs/inv_c_s$s train.seed=$s $G --variants $V > $L/inv_c_s$s.log 2>&1
done
# ---- 15E2: static model (seeds 0-2), scored on static clips; video models' per-frame scores for seeds 1-2
T1="$PY -u scripts/train.py --config configs/main_b1.yaml --set"
for s in 0 1 2; do
  [ -f runs/inv_static_s$s/DONE ] || run $T1 run_dir=runs/inv_static_s$s train.seed=$s $G data.static=true \
      > $L/inv_static_s$s.log 2>&1
  [ -f $O/static_model_s$s.json ] || run $PY -u scripts/probe_eval.py --run runs/inv_static_s$s --split holdout \
      --temporal static --per-frame --out $O/static_model_s$s.json > $L/static_model_s$s.log 2>&1
  if [ $s -gt 0 ]; then
    [ -f $O/t_orig_s$s.json ] || run $PY -u scripts/probe_eval.py --run runs/vitg_s$s --split holdout --per-frame \
        --out $O/t_orig_s$s.json > $L/t_orig_s$s.log 2>&1
  fi
done
# ---- 15F3: 384 pixels (seed 0; seeds 1-2 only if seed 0 gains at least 0.003 over runs/vitg_s0, i.e. >= 0.5896)
[ -f runs/inv_384_s0/DONE ] || run $T1 run_dir=runs/inv_384_s0 train.seed=0 model.backbone=vjepa2_g train.epochs=3 \
    train.warmup_epochs=0.25 "data.size=[384,384]" > $L/inv_384_s0.log 2>&1
B384=$($PY -c "import json;print(json.load(open('runs/inv_384_s0/DONE'))['best_holdout_cc'])")
if $PY -c "import sys;sys.exit(0 if $B384 >= 0.5896 else 1)"; then
  echo "$(date -Is) 384 seed 0 holdout $B384 >= 0.5896: seeds 1 and 2"
  for s in 1 2; do
    [ -f runs/inv_384_s$s/DONE ] || run $T1 run_dir=runs/inv_384_s$s train.seed=$s model.backbone=vjepa2_g train.epochs=3 \
        train.warmup_epochs=0.25 "data.size=[384,384]" > $L/inv_384_s$s.log 2>&1
  done
else
  echo "$(date -Is) 384 seed 0 holdout $B384 < 0.5896: single seed reported (rule 15F3)"
fi
for r in runs/inv_384_s0 runs/inv_384_s1 runs/inv_384_s2; do
  if [ -f $r/DONE ]; then
    b=$(basename $r)
    [ -f $O/holdout/$b.json ] || run $PY -u scripts/evaluate.py --run $r --split holdout --metrics cc,nss,sim,kld \
        --frame-step 5 --out $O/holdout/$b.json > $L/eval_$b.log 2>&1
  fi
done
# ---- 15A2: seed 0 retrained with the identical configuration
$PY - <<'EOF' || { echo "$(date -Is) A2 configuration differs from runs/vitg_s0; A2 not run"; exit 5; }
import json, sys
sys.path.insert(0, ".")
from surge.config import load_config
c = load_config("configs/main_b1.yaml", ["run_dir=runs/inv_repro_s0", "train.seed=0", "train.epochs=3",
                "train.warmup_epochs=0.25", "model.backbone=vjepa2_g", "data.size=[256,256]"])
r = json.load(open("runs/vitg_s0/config.json"))
c.pop("run_dir"); r.pop("run_dir")
c["model"] = {k: v for k, v in c["model"].items() if k in r["model"] or v not in (None, False)}
assert c == r, (c, r)
print("A2 configuration identical to runs/vitg_s0 (except run_dir)")
EOF
[ -f runs/inv_repro_s0/DONE ] || run $T1 run_dir=runs/inv_repro_s0 train.seed=0 $G > $L/inv_repro_s0.log 2>&1
# ---- 15I: zero-shot generalisation of runs/vitg_s0 (sigma = 8 x native width / 640, fixed in advance)
EV="$PY -u scripts/evaluate.py --run runs/vitg_s0 --metrics cc,nss,sim,kld"
zs() {  # name dataset split sigma frame_step
  [ -f $O/zs_$1.json ] || run $EV --dataset $2 --split $3 --sigma $4 --frame-step $5 --save-small $L/zs_small/$1 \
      --out $O/zs_$1.json > $L/zs_$1.log 2>&1
}
zs ucf ucf val 9 1
for F in diem_f1 coutrot1_f1 coutrot1_f2 coutrot1_f3 coutrot2_f1 coutrot2_f2 coutrot2_f3 etmd_f1 etmd_f2 etmd_f3 \
         avad_f1 avad_f2 avad_f3 summe_f1 summe_f2 summe_f3; do
  zs stavis_$F $F val 8 5
done
zs avimos avimos val 12 5
# MVVA through the probe evaluator (same protocol and metrics), keeping the maps for the multi-face analysis (15K)
[ -f $O/zs_mvva.json ] || run $PY -u scripts/probe_eval.py --run runs/vitg_s0 --dataset mvva --split val --sigma 8 \
    --frame-step 5 --save-small runs/inv/mvva_small_s0 --out $O/zs_mvva.json > $L/zs_mvva.log 2>&1
zs hollywood hollywood_vig val 7.5 5
# ---- 15M: dataset bias in zero-shot transfer (CPU; before A3, so that a failure of A3 cannot block it)
[ -f $O/summary_bias.json ] || run env CUDA_VISIBLE_DEVICES="" $PY -u scripts/inv_bias.py > $L/inv_bias.log 2>&1
# ---- 15A3: cost profile
[ -f $O/a3_cost.json ] || run $PY -u scripts/inv_cost.py --out $O/a3_cost.json > $L/a3_cost.log 2>&1
echo "$(date -Is) queue 15b done"
