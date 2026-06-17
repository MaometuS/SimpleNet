#!/usr/bin/env bash
set -euo pipefail

ratios=(${RATIOS:-0.02 0.05 0.10 0.20})
radius="${TSLRG_RADIUS:-1}"

for ratio in "${ratios[@]}"; do
  export RUN_NAME="exp_05_sparse_random_ratio${ratio}_r${radius}"
  export TSLRG_ANOMALY_MODE=anchored
  export TSLRG_RADIUS_MODE=anchor
  export TSLRG_RADIUS="$radius"
  export TSLRG_PATCH_MASK_MODE=random
  export TSLRG_PATCH_MASK_RATIO="$ratio"
  export TSLRG_REFINE_STEPS=0
  bash run.sh
done
