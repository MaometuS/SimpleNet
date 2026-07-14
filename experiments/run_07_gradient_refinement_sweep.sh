#!/usr/bin/env bash
set -euo pipefail

steps_list=(${REFINE_STEPS_LIST:-1 2 3})
step_sizes=(${REFINE_STEP_SIZES:-0.02 0.05 0.1})
ratio="${TSLRG_PATCH_MASK_RATIO:-0.10}"
block="${TSLRG_PATCH_MASK_BLOCK:-5}"
radius="${TSLRG_RADIUS:-1}"

for steps in "${steps_list[@]}"; do
  for step_size in "${step_sizes[@]}"; do
    export RUN_NAME="exp_07_refine_s${steps}_lr${step_size}_b${block}_ratio${ratio}_r${radius}_proj"
    export TSLRG_ANOMALY_MODE=anchored
    export TSLRG_RADIUS_MODE=anchor
    export TSLRG_RADIUS="$radius"
    export TSLRG_PATCH_MASK_MODE=block
    export TSLRG_PATCH_MASK_RATIO="$ratio"
    export TSLRG_PATCH_MASK_BLOCK="$block"
    export TSLRG_REFINE_STEPS="$steps"
    export TSLRG_REFINE_STEP_SIZE="$step_size"
    export TSLRG_PROJECT_FAKE_FEATS=1
    bash run.sh
  done
done
