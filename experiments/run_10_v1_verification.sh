#!/usr/bin/env bash
set -euo pipefail

# V1-backbone verification sequence (sequential, single GPU).
#
# Prerequisite (already in the working tree):
#   - backbones.py pins wideresnet50 to IMAGENET1K_V1
#   - simplenet.py defaults TSLRG_NOISE_SPACE=projected (upstream noise placement)
#
# Steps, in order:
#   1. Refit TSLRG k=512 on screw from V1 features -> runtime_quantiles_v1_k512
#      (fast; fail-early so the GPU-hours steps never run against a stale fit)
#   2. exp_00 SimpleNet-noise baseline on screw. Expected ~0.98 I-AUROC
#      (V2 run gave 0.9317; Original V1 run gave 0.9818).
#   3. k512 winner configs (q0.99, delta 0 and 0.25) against the V1 checkpoint.
#
# Each step is skipped if its results already exist, so the script is
# resumable after an interruption. Force a redo by deleting the result dir
# (or the checkpoint for step 1).
#
# Usage:
#   bash experiments/run_10_v1_verification.sh
# Env overrides: DATAPATH, GPU, CLASSNAME (default screw).

cd "$(dirname "$0")/.."

classname="${CLASSNAME:-screw}"
export CLASSNAME="$classname"
export GPU="${GPU:-0}"

k=512
v1_ckpt_dir="true_spatial_low_rank_gaussian/runtime_quantiles_v1_k${k}"
results_base="results/MVTecAD_Results/simplenet_mvtec"
log_dir="logs/v1_verification"
mkdir -p "$log_dir"

banner() { printf '\n============ %s ============\n' "$*"; }

# --- Step 1: refit TSLRG on V1 features --------------------------------------
ckpt_file="${v1_ckpt_dir}/mvtec_${classname}.pt"
if [[ -f "$ckpt_file" ]]; then
  banner "STEP 1 SKIPPED: $ckpt_file already exists"
else
  banner "STEP 1: refitting TSLRG k=${k} on ${classname} (V1 features)"
  TSLRG_K="$k" \
  TSLRG_CHECKPOINT_DIR="$v1_ckpt_dir" \
  bash low_rank_gaussian.sh 2>&1 | tee "${log_dir}/step1_tslrg_fit.log"
  [[ -f "$ckpt_file" ]] || { echo "FATAL: expected checkpoint $ckpt_file was not written"; exit 1; }
fi

# --- Step 2: exp_00 baseline on V1 -------------------------------------------
baseline_run="exp_00_simplenet_noise_v1"
if [[ -f "${results_base}/${baseline_run}/results.csv" ]]; then
  banner "STEP 2 SKIPPED: ${baseline_run} results exist"
else
  banner "STEP 2: exp_00 SimpleNet-noise baseline on ${classname} (V1)"
  RUN_NAME="$baseline_run" \
  TSLRG_NOISE_SPACE=projected \
  TSLRG_CHECKPOINT_DIR="$v1_ckpt_dir" \
  bash experiments/run_00_simplenet_noise.sh 2>&1 | tee "${log_dir}/step2_exp00_v1.log"
fi

# --- Step 3: k512 winner configs on V1 ---------------------------------------
run_prefix="exp_10_v1_k${k}"
quantiles="0.99"
deltas="0 0.25"
missing=0
for q in $quantiles; do for d in $deltas; do
  [[ -f "${results_base}/${run_prefix}_q${q}_delta${d}/results.csv" ]] || missing=1
done; done
if [[ "$missing" == "0" ]]; then
  banner "STEP 3 SKIPPED: all ${run_prefix} results exist"
else
  banner "STEP 3: k512 winners (q=${quantiles}, delta=${deltas}) on V1 checkpoint"
  QUANTILES="$quantiles" \
  DELTAS="$deltas" \
  TSLRG_EXPECTED_K="$k" \
  TSLRG_CHECKPOINT_DIR="$v1_ckpt_dir" \
  RUN_PREFIX="$run_prefix" \
  bash experiments/run_09_default_preprojected_quantile_delta_sweep.sh 2>&1 | tee "${log_dir}/step3_k512_winners.log"
fi

# --- Summary ------------------------------------------------------------------
banner "SUMMARY (${classname})"
echo "Reference points:"
echo "  Original (V1, upstream code):   0.9818 / 0.9929 / 0.9612"
echo "  exp_00 on V2 (broken baseline): 0.9317 / 0.9175 / 0.7733"
echo
for run in "$baseline_run" $(for q in $quantiles; do for d in $deltas; do echo "${run_prefix}_q${q}_delta${d}"; done; done); do
  csv="${results_base}/${run}/results.csv"
  if [[ -f "$csv" ]]; then
    echo "--- $run"
    column -s, -t < "$csv"
  else
    echo "--- $run: MISSING"
  fi
  echo
done
