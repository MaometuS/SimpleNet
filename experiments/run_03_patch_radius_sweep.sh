#!/usr/bin/env bash
set -euo pipefail

radii=(${RADII:-0.25 0.5 1 2 5})

for radius in "${radii[@]}"; do
  export RUN_NAME="exp_03_patch_radius_r${radius}_proj"
  export TSLRG_ANOMALY_MODE=anchored
  export TSLRG_RADIUS_MODE=patch
  export TSLRG_RADIUS="$radius"
  export TSLRG_PATCH_MASK_MODE=all
  export TSLRG_REFINE_STEPS=0
  export TSLRG_PROJECT_FAKE_FEATS=1
  bash run.sh
done
