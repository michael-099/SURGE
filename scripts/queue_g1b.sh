#!/usr/bin/env bash
# Waits for chain_g1a to end, then runs G1 for V-JEPA 2.1-L, then the G2 surprise test (one GPU job at a time).
cd "$(dirname "$0")/.." || exit 1
until grep -qE "chain finished|chain stops" runs/chain_g1a.log 2>/dev/null; do sleep 30; done
grep -q "chain finished" runs/chain_g1a.log || { echo "$(date -Is) g1a did not finish cleanly; not continuing"; exit 1; }
bash scripts/chain.sh configs/g1_vjepa21_l.yaml || exit $?
[ -f SAFETY_STOP.txt ] && { echo "$(date -Is) SAFETY_STOP present; G2 not started"; exit 2; }
echo "$(date -Is) start G2"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -u scripts/g2_surprise.py > runs/g2_stdout.log 2>&1
echo "$(date -Is) G2 rc=$?"
