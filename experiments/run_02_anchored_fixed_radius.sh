#!/usr/bin/env bash
set -euo pipefail

radius="${TSLRG_RADIUS:-1}"

export RUN_NAME="${RUN_NAME:-exp_02_anchored_fixed_radius_r${radius}}"
export TSLRG_ANOMALY_MODE=anchored
export TSLRG_RADIUS_MODE=threshold
export TSLRG_RADIUS="$radius"
export TSLRG_PATCH_MASK_MODE=all
export TSLRG_REFINE_STEPS=0

bash run.sh
