#!/usr/bin/env bash
set -euo pipefail

blocks=(${BLOCKS:-3 5 7 9})
ratio="${TSLRG_PATCH_MASK_RATIO:-0.10}"
radius="${TSLRG_RADIUS:-1}"

for block in "${blocks[@]}"; do
  export RUN_NAME="exp_06_sparse_block_b${block}_ratio${ratio}_r${radius}_1"
  export TSLRG_ANOMALY_MODE=anchored
  export TSLRG_RADIUS_MODE=anchor
  export TSLRG_RADIUS="$radius"
  export TSLRG_PATCH_MASK_MODE=block
  export TSLRG_PATCH_MASK_RATIO="$ratio"
  export TSLRG_PATCH_MASK_BLOCK="$block"
  export TSLRG_REFINE_STEPS=0
  export TSLRG_PROJECT_FAKE_FEATS=1
  bash run.sh
done
