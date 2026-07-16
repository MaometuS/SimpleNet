#!/usr/bin/env bash
set -euo pipefail

# Experiment 9 keeps Experiment 8 fixed and sweeps the anomaly shell:
#   r = sqrt(T_p(q)) + delta * U(0, 1)
# The checkpoint must include calibration scores. Regenerate it once with:
#   TSLRG_K=512 CLASSNAME=screw bash low_rank_gaussian.sh
quantiles=(${QUANTILES:-0.90 0.95 0.99 0.995})
deltas=(${DELTAS:-0 0.25 0.5 1 2})
expected_k="${TSLRG_EXPECTED_K:-512}"
checkpoint_dir="${TSLRG_CHECKPOINT_DIR:-true_spatial_low_rank_gaussian/runtime_quantiles_v2_k${expected_k}}"
run_prefix="${RUN_PREFIX:-exp_09_default_preprojected_k${expected_k}}"

for quantile in "${quantiles[@]}"; do
  for delta in "${deltas[@]}"; do
    TSLRG_QUANTILE="$quantile" \
    TSLRG_DELTA="$delta" \
    TSLRG_EXPECTED_K="$expected_k" \
    TSLRG_CHECKPOINT_DIR="$checkpoint_dir" \
    RUN_NAME="${run_prefix}_q${quantile}_delta${delta}" \
    bash experiments/run_08_default_preprojected_fake.sh
  done
done
