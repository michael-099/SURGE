#!/usr/bin/env bash
# research/log.md sections 13B, 13C and 13D (with the addenda of 2026-11-11):
#   1. token maps of the prediction error with both mask tokens (GPU, zero-shot, no training)
#   2. the incremental-information analysis and the G2 re-score (CPU)
#   3. measured inference time (GPU, last, nothing else running)
# Run after queue D. One GPU job at a time. Resumable (g2_maps.py skips finished videos).
# Stops if SAFETY_STOP.txt exists or a GPU step fails.
# Usage: bash scripts/queue_review_e.sh > runs/review/queue_e.log 2>&1
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=.venv/Scripts/python.exe
stop() { [ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; queue stops"; exit 2; }; }
O=runs/review; mkdir -p $O runs/g2/maps

stop; echo "$(date -Is) g2_maps"
$PY -u scripts/g2_maps.py --out runs/g2/maps > $O/g2_maps.log 2>&1 || { echo "$(date -Is) FAILED: g2_maps"; exit 3; }

stop; echo "$(date -Is) g2_incremental"
$PY -u scripts/g2_incremental.py --out $O/g2_incremental.json > $O/g2_incremental.txt 2>&1
echo "$(date -Is) g2_incremental exit $?"   # 4 = the reproduction check failed; the timing below does not depend on it

# SalFoM's repository is third-party code: isolated mode, its path appended last (see scripts/time_inference.py)
stop; echo "$(date -Is) time_inference"
$PY -I -u scripts/time_inference.py --out $O/timing.json > $O/timing.txt 2>&1 || { echo "$(date -Is) FAILED: time_inference"; exit 3; }
echo "$(date -Is) queue E done"
