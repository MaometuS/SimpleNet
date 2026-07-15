#!/usr/bin/env bash
set -euo pipefail

# Experiment 9 keeps Experiment 8 fixed and changes only the width of the
# anomaly shell: r = sqrt(T_p) + delta * U(0, 1). Delta 1 is the Exp. 8 baseline.
deltas=(${DELTAS:-0 0.25 0.5 1 2})
run_prefix="${RUN_PREFIX:-exp_09_default_preprojected}"

for delta in "${deltas[@]}"; do
  TSLRG_DELTA="$delta" \
  RUN_NAME="${run_prefix}_delta${delta}" \
  bash experiments/run_08_default_preprojected_fake.sh
done
