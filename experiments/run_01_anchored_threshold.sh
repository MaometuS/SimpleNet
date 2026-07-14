#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="${RUN_NAME:-exp_01_anchored_threshold_proj}"
export TSLRG_ANOMALY_MODE=anchored
export TSLRG_RADIUS_MODE=threshold
unset TSLRG_RADIUS || true
export TSLRG_PATCH_MASK_MODE=all
export TSLRG_PROJECT_FAKE_FEATS=1
export TSLRG_REFINE_STEPS=0

bash run.sh
