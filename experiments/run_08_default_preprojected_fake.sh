#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="${RUN_NAME:-exp_08_default_preprojected_fake}"

# Generate anomalies with the original full-sphere TSLRG formulation, then
# transform them with the same trainable projection used for real features.
export PRE_PROJ="${PRE_PROJ:-1}"
export TSLRG_ANOMALY_MODE=default
export TSLRG_PROJECT_FAKE_FEATS=1
export TSLRG_PATCH_MASK_MODE=all
export TSLRG_REFINE_STEPS=0
unset TSLRG_RADIUS || true
unset TSLRG_RADIUS_MODE || true

bash run.sh
