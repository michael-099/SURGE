#!/usr/bin/env bash
# Main program (research/log.md sections 4-5): waits for the G1/G2 queue, then B1 x3 seeds, M x3 seeds, ablations.
# Backbone and schedule come from the G1 decision written in the log (passed here as overrides).
cd "$(dirname "$0")/.." || exit 1
BB="${BB:-vjepa2_l}"; SZ="${SZ:-[256,256]}"; EP="${EP:-3}"
until grep -qE "G2 rc=|not continuing|G2 not started|chain stops" runs/chain_g1b.log 2>/dev/null; do sleep 30; done
C=()
for s in 0 1 2; do
  C+=("configs/main_b1.yaml::run_dir=runs/main_b1_s$s train.seed=$s train.epochs=$EP train.warmup_epochs=0.25 model.backbone=$BB data.size=$SZ")
done
for s in 0 1 2; do
  C+=("configs/main_m.yaml::run_dir=runs/main_m_s$s train.seed=$s train.epochs=$EP model.backbone=$BB data.size=$SZ train.init_from=runs/main_b1_s$s/best.pt")
done
A="train.seed=0 train.epochs=$EP train.warmup_epochs=0.25 model.backbone=$BB data.size=$SZ"
C+=("configs/main_b1.yaml::run_dir=runs/abl_noprior $A model.prior=false")
C+=("configs/main_b1.yaml::run_dir=runs/abl_noaux $A model.aux=false")
C+=("configs/main_b1.yaml::run_dir=runs/abl_stride4 $A data.stride=4")
C+=("configs/main_b1.yaml::run_dir=runs/abl_salicon $A data.salicon=true")
bash scripts/chain.sh "${C[@]}"
