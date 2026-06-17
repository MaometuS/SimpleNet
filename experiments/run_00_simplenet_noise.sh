#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="${RUN_NAME:-exp_00_simplenet_noise}"
export TSLRG_ANOMALY_MODE=simplenet_noise
export TSLRG_PATCH_MASK_MODE=all
export TSLRG_REFINE_STEPS=0

bash run.sh
