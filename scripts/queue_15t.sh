#!/usr/bin/env bash
# Section 15 (research/log.md), test-time block on the DHF1K holdout with runs/vitg_s0: the evaluator check, then
# E1 (temporal order), E3 (stride), G (tilings), F2 (input size) and H (robustness). Waits for A1 to finish (one GPU
# job at a time). Stops if SAFETY_STOP.txt (repo root) exists or if the evaluator check fails. Resumable (skips
# finished outputs). Usage: bash scripts/queue_15t.sh > runs/inv/queue_15t.log 2>&1
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
O=runs/review/inv
L=runs/inv
mkdir -p $O $L
stop() { if [ -f SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; fi; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
n=0
until [ -f $O/a1_val_s0.json ]; do
  n=$((n+1)); if [ $n -gt 180 ]; then echo "$(date -Is) A1 not finished after 3 h; not starting"; exit 1; fi
  sleep 60
done
sleep 20  # let A1's process release the GPU
echo "$(date -Is) A1 done"
P="$PY -u scripts/probe_eval.py --run runs/vitg_s0 --split holdout"
probe() {  # name, extra args
  local N=$1; shift
  [ -f $O/$N.json ] || run $P "$@" --out $O/$N.json > $L/$N.log 2>&1
}
# evaluator check (section 15A): default settings must reproduce runs/review/holdout/vitg_s0.json per video
probe t_orig_s0 --per-frame --positions --flip-consistency
$PY - <<'EOF' || { echo "$(date -Is) evaluator check FAILED; queue stops"; exit 4; }
import json, sys
a = json.load(open("runs/review/inv/t_orig_s0.json"))["per_video"]
b = json.load(open("runs/review/holdout/vitg_s0.json"))["per_video"]
assert set(a) == set(b), "video sets differ"
worst = max(abs(a[v][k] - b[v][k]) for v in a for k in ("cc", "nss", "sim", "kld"))
print(f"evaluator check: max per-video difference {worst:.2e}")
sys.exit(0 if worst < 1e-4 else 1)
EOF
echo "$(date -Is) evaluator check passed"
# E1: temporal order (seed 0)
probe t_reverse_s0 --temporal reverse --per-frame
probe t_shuffle_s0 --temporal shuffle --per-frame
probe t_static_s0 --temporal static --per-frame
# E3: temporal stride at test time
probe t_stride1_s0 --stride 1
probe t_stride4_s0 --stride 4
# G: tilings of the multi-frame output
probe t_all16_s0 --tiling all16
probe t_central4_s0 --tiling central4
probe t_overlap_s0 --tiling overlap
probe t_dense_s0 --tiling dense --per-frame
# F2: input size at test time (encoder positions rebuilt for each size)
for S in 192 224 320 384; do probe t_size${S}_s0 --size $S; done
# H: robustness
probe t_noop_s0 --perturb noop
probe t_noise04_s0 --perturb noise:0.04
probe t_noise16_s0 --perturb noise:0.16
probe t_blur2_s0 --perturb blur:2
probe t_blur8_s0 --perturb blur:8
probe t_jpeg10_s0 --perturb jpeg:10
probe t_contrast05_s0 --perturb contrast:0.5
probe t_bright05_s0 --perturb bright:0.5
probe t_gray_s0 --perturb gray
probe t_dup_s0 --perturb dup
echo "$(date -Is) queue 15t done"
