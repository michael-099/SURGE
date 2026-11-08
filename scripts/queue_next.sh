#!/usr/bin/env bash
# research/log.md section 11, one GPU job at a time, in order 11A, 11B, 11C, 11D, 11E. Resumable.
# Stops if SAFETY_STOP.txt exists or a step fails. Usage: bash scripts/queue_next.sh > runs/queue_next.log 2>&1
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; }
run() { stop; echo "$(date -Is) $*"; "$@" || { echo "$(date -Is) FAILED: $*"; exit 3; }; }
sig() { $PY -c "import json;print(json.load(open('$1'))['best_sigma'])"; }
best() { $PY -c "import json;d=[json.loads(l) for l in open('$1/log.jsonl')];print([x for x in d if x['event']=='done'][-1]['best'])"; }
SZ="data.size=[256,256]"
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
cmp() { [ -f $3 ] || $PY scripts/compare.py $1 $2 --out ${3%.txt}.json > $3; }

# ------------------------------------------------------------------ 11A train on 600
O=runs/vitg600; mkdir -p $O
for s in 0 1 2; do
  train runs/vitg600_s$s --config configs/main_b1.yaml --set run_dir=runs/vitg600_s$s train.seed=$s train.epochs=3 \
      train.warmup_epochs=0.25 model.backbone=vjepa2_g "$SZ" data.train_split=configs/splits/dhf1k_train600.txt eval.frame_step=25
done
E600="--run runs/vitg600_s0 --run runs/vitg600_s1 --run runs/vitg600_s2"
[ -f $O/val_single_s0.json ] || run $PY -u scripts/evaluate.py --run runs/vitg600_s0 --ckpt epoch_1.pt --split val --metrics all \
    --sigma 8 --out $O/val_single_s0.json > $O/val_single_s0.log 2>&1
[ -f $O/val_ens3.json ] || run $PY -u scripts/evaluate.py $E600 --ckpt epoch_1.pt --split val --metrics all --sigma 8 \
    --out $O/val_ens3.json > $O/val_ens3.log 2>&1
cmp $O/val_single_s0.json runs/vitg/val_single_s0.json $O/cmp_single_vs_540.txt
cmp $O/val_ens3.json runs/vitg/val_ens3.json $O/cmp_ens3_vs_540.txt
cmp $O/val_ens3.json runs/metric_check/salfom_val_all.json $O/cmp_ens3_vs_salfom.txt
USE600=$($PY -c "
import re
t=open('runs/vitg600/cmp_ens3_vs_540.txt').read().splitlines()
r={l.split()[0]:l for l in t if l.strip()}
lo=float(re.search(r'\[([-+0-9.]+),',r['cc']).group(1))
worse=0
for k,l in r.items():
    d=float(re.search(r'A-B ([-+0-9.]+)',l).group(1))
    worse+= (d>0) if k=='kld' else (d<0)
print(int(lo>0 and worse<=2))")
echo "$(date -Is) 11A decision: use600=$USE600"
echo $USE600 > $O/use600.txt

# ------------------------------------------------------------------ 11B DHF1K test maps
SUB=D:/DHF1K_submission; mkdir -p runs/dhf1k_submission_vitg
if [ "$USE600" = 1 ]; then ENS="$E600"; CK=epoch_1.pt; REF=$O/val_ens3.json
else ENS="--run runs/vitg_s0 --run runs/vitg_s1 --run runs/vitg_s2"; CK=best.pt; REF=runs/vitg/val_ens3.json; fi
if [ ! -f runs/dhf1k_submission_vitg/format_check.ok ]; then
  run $PY -u scripts/predict_dhf1k_test.py $ENS --ckpt $CK --sigma 8 --first 601 --last 605 --out $SUB/format_check_vitg
  run $PY -u scripts/evaluate.py --maps $SUB/format_check_vitg --split val --max-videos 5 --metrics cc,nss \
      --out runs/dhf1k_submission_vitg/format_check.json
  export REF; run $PY - <<'PY'
import json, os
a = json.load(open("runs/dhf1k_submission_vitg/format_check.json"))["per_video"]
b = json.load(open(os.environ["REF"]))["per_video"]
d = {k: abs(a[k]["cc"] - b[k]["cc"]) for k in a}
print("per-video |dCC|:", {k: round(v, 5) for k, v in d.items()})
assert len(d) == 5 and max(d.values()) <= 0.002, "format check failed"
open("runs/dhf1k_submission_vitg/format_check.ok", "w").write(json.dumps(d))
PY
fi
run $PY -u scripts/predict_dhf1k_test.py $ENS --ckpt $CK --sigma 8 --out $SUB/SURGE-D-vitg > runs/dhf1k_submission_vitg/predict.log 2>&1
[ -f $SUB/SURGE-D-vitg.zip ] || run $PY - <<'PY'
import zipfile, hashlib
from pathlib import Path
src = Path("D:/DHF1K_submission/SURGE-D-vitg"); dst = Path("D:/DHF1K_submission/SURGE-D-vitg.zip")
dirs = sorted(p for p in src.iterdir() if p.is_dir()); n = 0
assert len(dirs) == 300, len(dirs)
with zipfile.ZipFile(str(dst) + ".tmp", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for d in dirs:
        for f in sorted(d.glob("*.png")):
            z.write(f, f"SURGE-D/{d.name}/{f.name}"); n += 1
Path(str(dst) + ".tmp").replace(dst)
with zipfile.ZipFile(dst) as z:
    assert z.testzip() is None
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
print("zip", dst, n, "files", dst.stat().st_size, "bytes sha256", h)
open("runs/dhf1k_submission_vitg/zip.txt", "w").write(f"{n} {dst.stat().st_size} {h}\n")
PY
echo "$(date -Is) 11B done"

# ------------------------------------------------------------------ 11C 32-frame clips
O=runs/vitg_t32; mkdir -p $O
T32="data.T=32 model.backbone=vjepa2_g $SZ train.epochs=3 train.warmup_epochs=0.25"
train runs/vitg_t32_s0 --config configs/main_b1.yaml --set run_dir=runs/vitg_t32_s0 train.seed=0 $T32
G=$($PY -c "print(int($(best runs/vitg_t32_s0) >= 0.5904))")
if [ "$G" = 1 ]; then
  echo "$(date -Is) T32 gate PASSED ($(best runs/vitg_t32_s0))"
  for s in 1 2; do
    train runs/vitg_t32_s$s --config configs/main_b1.yaml --set run_dir=runs/vitg_t32_s$s train.seed=$s $T32
  done
  E32="--run runs/vitg_t32_s0 --run runs/vitg_t32_s1 --run runs/vitg_t32_s2"
  [ -f $O/sigma_single.json ] || run $PY -u scripts/tune_sigma.py --run runs/vitg_t32_s0 --out $O/sigma_single.json > $O/sigma_single.log 2>&1
  [ -f $O/sigma_ens3.json ] || run $PY -u scripts/tune_sigma.py $E32 --out $O/sigma_ens3.json > $O/sigma_ens3.log 2>&1
  [ -f $O/val_single_s0.json ] || run $PY -u scripts/evaluate.py --run runs/vitg_t32_s0 --split val --metrics all \
      --sigma $(sig $O/sigma_single.json) --out $O/val_single_s0.json > $O/val_single_s0.log 2>&1
  [ -f $O/val_ens3.json ] || run $PY -u scripts/evaluate.py $E32 --split val --metrics all \
      --sigma $(sig $O/sigma_ens3.json) --out $O/val_ens3.json > $O/val_ens3.log 2>&1
  cmp $O/val_single_s0.json runs/vitg/val_single_s0.json $O/cmp_single_vs_t16.txt
  cmp $O/val_ens3.json runs/vitg/val_ens3.json $O/cmp_ens3_vs_t16.txt
else
  echo "$(date -Is) T32 gate FAILED ($(best runs/vitg_t32_s0) < 0.5904); val not touched"
fi
echo "$(date -Is) 11C done"

# ------------------------------------------------------------------ 11D audio (scripts/queue_audio.sh, written separately)
until [ -f scripts/queue_audio.ready ]; do stop; sleep 120; done
bash scripts/queue_audio.sh || { echo "$(date -Is) 11D failed; queue stops"; exit 3; }
echo "$(date -Is) 11D done"

# ------------------------------------------------------------------ 11E ViT-g on UCF, STAViS, Hollywood
O=runs/g_other; mkdir -p $O
G="model.backbone=vjepa2_g $SZ train.stage=1 train.epochs=3 train.warmup_epochs=0.25"
for k in 0 1 2; do
  train runs/g_ucf_s$k --config configs/ucf_ft.yaml --set run_dir=runs/g_ucf_s$k train.seed=$k train.init_from=runs/vitg_s$k/best.pt $G
done
U="--run runs/g_ucf_s0 --run runs/g_ucf_s1 --run runs/g_ucf_s2"
[ -f $O/ucf_sigma_s0.json ] || run $PY -u scripts/tune_sigma.py --run runs/g_ucf_s0 --dataset ucf --split holdout --frame-step 1 --out $O/ucf_sigma_s0.json > $O/ucf_sigma_s0.log 2>&1
[ -f $O/ucf_sigma_ens.json ] || run $PY -u scripts/tune_sigma.py $U --dataset ucf --split holdout --frame-step 1 --out $O/ucf_sigma_ens.json > $O/ucf_sigma_ens.log 2>&1
[ -f $O/ucf_s0.json ] || run $PY -u scripts/evaluate.py --run runs/g_ucf_s0 --dataset ucf --split val --metrics all --sigma $(sig $O/ucf_sigma_s0.json) --out $O/ucf_s0.json > $O/ucf_s0.log 2>&1
[ -f $O/ucf_ens.json ] || run $PY -u scripts/evaluate.py $U --dataset ucf --split val --metrics all --sigma $(sig $O/ucf_sigma_ens.json) --out $O/ucf_ens.json > $O/ucf_ens.log 2>&1
echo "$(date -Is) 11E UCF done"

FOLDS="diem_f1 coutrot1_f1 coutrot1_f2 coutrot1_f3 coutrot2_f1 coutrot2_f2 coutrot2_f3 etmd_f1 etmd_f2 etmd_f3 avad_f1 avad_f2 avad_f3 summe_f1 summe_f2 summe_f3"
for F in $FOLDS; do
  NTR=$($PY -c "import sys;sys.path.insert(0,'.');from surge.data import stavis_split_videos as s;print(len(s('$F','train','configs/splits')))")
  CPV=$(( (1500 + NTR - 1) / NTR ))
  FS=2; [ "$F" = diem_f1 ] && FS=5
  for k in 0 1 2; do
    train runs/g_stavis_${F}_s$k --config configs/ucf_ft.yaml --set run_dir=runs/g_stavis_${F}_s$k train.seed=$k \
        data.dataset=$F data.root=D:/Datasets train.init_from=runs/vitg_s$k/best.pt train.clips_per_video=$CPV eval.frame_step=$FS $G
  done
  M="--run runs/g_stavis_${F}_s0 --run runs/g_stavis_${F}_s1 --run runs/g_stavis_${F}_s2"
  [ -f $O/stavis_sigma_s0_$F.json ] || run $PY -u scripts/tune_sigma.py --run runs/g_stavis_${F}_s0 --dataset $F --split holdout --frame-step $FS --out $O/stavis_sigma_s0_$F.json > $O/stavis_sigma_s0_$F.log 2>&1
  [ -f $O/stavis_sigma_ens_$F.json ] || run $PY -u scripts/tune_sigma.py $M --dataset $F --split holdout --frame-step $FS --out $O/stavis_sigma_ens_$F.json > $O/stavis_sigma_ens_$F.log 2>&1
  [ -f $O/stavis_s0_$F.json ] || run $PY -u scripts/evaluate.py --run runs/g_stavis_${F}_s0 --dataset $F --split val --metrics all --sigma $(sig $O/stavis_sigma_s0_$F.json) --out $O/stavis_s0_$F.json > $O/stavis_s0_$F.log 2>&1
  [ -f $O/stavis_ens_$F.json ] || run $PY -u scripts/evaluate.py $M --dataset $F --split val --metrics all --sigma $(sig $O/stavis_sigma_ens_$F.json) --out $O/stavis_ens_$F.json > $O/stavis_ens_$F.log 2>&1
done
echo "$(date -Is) 11E STAViS done"

for k in 0 1 2; do
  train runs/g_hollywood_s$k --config configs/ucf_ft.yaml --set run_dir=runs/g_hollywood_s$k train.seed=$k \
      data.dataset=hollywood_vig train.init_from=runs/vitg_s$k/best.pt train.clips_per_video=3 eval.frame_step=5 $G
done
H="--run runs/g_hollywood_s0 --run runs/g_hollywood_s1 --run runs/g_hollywood_s2"
[ -f $O/hollywood_sigma_s0.json ] || run $PY -u scripts/tune_sigma.py --run runs/g_hollywood_s0 --dataset hollywood_vig --split holdout --frame-step 5 --out $O/hollywood_sigma_s0.json > $O/hollywood_sigma_s0.log 2>&1
[ -f $O/hollywood_sigma_ens.json ] || run $PY -u scripts/tune_sigma.py $H --dataset hollywood_vig --split holdout --frame-step 5 --out $O/hollywood_sigma_ens.json > $O/hollywood_sigma_ens.log 2>&1
[ -f $O/hollywood_s0.json ] || run $PY -u scripts/evaluate.py --run runs/g_hollywood_s0 --dataset hollywood_vig --split val --metrics all --sigma $(sig $O/hollywood_sigma_s0.json) --out $O/hollywood_s0.json > $O/hollywood_s0.log 2>&1
[ -f $O/hollywood_ens.json ] || run $PY -u scripts/evaluate.py $H --dataset hollywood_vig --split val --metrics all --sigma $(sig $O/hollywood_sigma_ens.json) --out $O/hollywood_ens.json > $O/hollywood_ens.log 2>&1
echo "$(date -Is) queue_next finished"
