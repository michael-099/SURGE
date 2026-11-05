#!/usr/bin/env bash
# Run SURGE training configs one after another (one GPU job at a time).
# Skips runs that already have a DONE file; stops if SAFETY_STOP.txt exists or a run fails.
# Usage: bash scripts/chain.sh configs/a.yaml configs/b.yaml ...   (each argument may carry --set overrides after '::')
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
for spec in "$@"; do
  cfg="${spec%%::*}"; extra=""; [[ "$spec" == *::* ]] && extra="${spec#*::}"
  if [ -f SAFETY_STOP.txt ]; then echo "$(date -Is) SAFETY_STOP.txt present; chain stops before $cfg"; exit 2; fi
  run=$(.venv/Scripts/python.exe -c "import sys;sys.path.insert(0,'.');from surge.config import load_config;print(load_config('$cfg', '$extra'.split())['run_dir'])")
  if [ -f "$run/DONE" ]; then echo "$(date -Is) skip $cfg ($run/DONE exists)"; continue; fi
  mkdir -p "$run"
  echo "$(date -Is) start $cfg $extra -> $run"
  .venv/Scripts/python.exe -u scripts/train.py --config "$cfg" --set $extra > "$run/stdout.log" 2>&1
  rc=$?
  echo "$(date -Is) end $cfg rc=$rc"
  if [ $rc -ne 0 ] || [ ! -f "$run/DONE" ]; then echo "$(date -Is) chain stops (rc=$rc or no DONE)"; exit 3; fi
done
echo "$(date -Is) chain finished"
