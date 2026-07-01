#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="${RUN_NAME:-exp_00_simplenet_noise}"

# Match the upstream SimpleNet run.sh baseline values.
export NOISE_STD="${NOISE_STD:-0.015}"
export DSC_MARGIN="${DSC_MARGIN:-0.5}"
export PRE_PROJ="${PRE_PROJ:-1}"
export MIX_NOISE="${MIX_NOISE:-1}"

export TSLRG_ANOMALY_MODE=simplenet_noise
export TSLRG_PATCH_MASK_MODE=all
export TSLRG_REFINE_STEPS=0
unset TSLRG_RADIUS || true
unset TSLRG_RADIUS_MODE || true

bash run.sh
