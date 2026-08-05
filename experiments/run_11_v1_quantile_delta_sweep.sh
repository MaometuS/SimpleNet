#!/usr/bin/env bash
set -euo pipefail

# Experiment 11: full quantile x delta sweep on the V1 backbone.
#
# Motivation: the q=0.99 "winner" used in exp_10 was selected on the V2
# landscape (exp_09). The two V1 cells measured so far flipped the delta
# ordering (delta 0.25 > delta 0 by +1.5 I-AUROC), which suggests the V1
# optimum sits at a larger radius and the whole grid must be re-measured.
# Same grid as exp_09 so every cell has a direct V2 counterpart.
#
# Prerequisites (already in the working tree):
#   - backbones.py pins wideresnet50 to IMAGENET1K_V1 (guarded below)
#   - TSLRG k=512 checkpoint fitted on V1 features (refit here if missing)
#
# The two cells already measured by run_10 (q0.99, delta 0 / 0.25) are
# reused from their exp_10_v1_k512_* result dirs, not re-run: they were
# produced by the identical run_08 path against the same V1 checkpoint.
# 18 of 20 cells run fresh. Each cell is skipped if its results.csv
# exists, so the sweep is resumable after an interruption.
#
# Usage:
#   bash experiments/run_11_v1_quantile_delta_sweep.sh
# Env overrides: DATAPATH, GPU, CLASSNAME (default screw),
#   QUANTILES, DELTAS, RUN_PREFIX.

cd "$(dirname "$0")/.."

classname="${CLASSNAME:-screw}"
export CLASSNAME="$classname"
export GPU="${GPU:-0}"

k=512
quantiles=(${QUANTILES:-0.90 0.95 0.99 0.995})
deltas=(${DELTAS:-0 0.25 0.5 1 2})
run_prefix="${RUN_PREFIX:-exp_11_v1_k${k}}"
reuse_prefix="exp_10_v1_k${k}"
v1_ckpt_dir="true_spatial_low_rank_gaussian/runtime_quantiles_v1_k${k}"
results_base="results/MVTecAD_Results/simplenet_mvtec"
v2_prefix="exp_09_default_preprojected_k${k}"
log_dir="logs/v1_sweep"
mkdir -p "$log_dir"

banner() { printf '\n============ %s ============\n' "$*"; }

# Returns (echoes) the existing results.csv for a cell, preferring the
# exp_11 dir, falling back to the identically-configured exp_10 run.
existing_csv() {
  local q="$1" d="$2" csv
  for prefix in "$run_prefix" "$reuse_prefix"; do
    csv="${results_base}/${prefix}_q${q}_delta${d}/results.csv"
    [[ -f "$csv" ]] && { echo "$csv"; return 0; }
  done
  return 1
}

# --- Guard: backbone must still be pinned to V1 ------------------------------
if ! grep -q 'Wide_ResNet50_2_Weights.IMAGENET1K_V1' backbones.py; then
  echo "FATAL: backbones.py no longer pins wideresnet50 to IMAGENET1K_V1."
  echo "This sweep is meaningless against V2 weights. Aborting."
  exit 1
fi

# --- Step 1: ensure V1-fitted TSLRG checkpoint -------------------------------
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

# --- Step 2: sweep -----------------------------------------------------------
total=$(( ${#quantiles[@]} * ${#deltas[@]} ))
done_count=0
for q in "${quantiles[@]}"; do
  for d in "${deltas[@]}"; do
    done_count=$(( done_count + 1 ))
    if csv=$(existing_csv "$q" "$d"); then
      banner "CELL ${done_count}/${total} SKIPPED: q=${q} delta=${d} (${csv})"
      continue
    fi
    run_name="${run_prefix}_q${q}_delta${d}"
    banner "CELL ${done_count}/${total}: q=${q} delta=${d} -> ${run_name}"
    TSLRG_QUANTILE="$q" \
    TSLRG_DELTA="$d" \
    TSLRG_EXPECTED_K="$k" \
    TSLRG_CHECKPOINT_DIR="$v1_ckpt_dir" \
    RUN_NAME="$run_name" \
    bash experiments/run_08_default_preprojected_fake.sh 2>&1 | tee "${log_dir}/${run_name}.log"
  done
done

# --- Summary: V1 cell next to its V2 (exp_09) counterpart --------------------
banner "SUMMARY (${classname}) - instance/full-pixel/anomaly-pixel AUROC"
echo "Reference points (V1 backbone):"
echo "  Original (upstream):        0.9818 / 0.9929 / 0.9612"
echo "  exp_00 baseline (V1 rerun): 0.9824 / 0.9926 / 0.9685"
echo
printf '%-18s %-30s %-30s\n' "cell" "V1 (this sweep)" "V2 (exp_09)"
for q in "${quantiles[@]}"; do
  for d in "${deltas[@]}"; do
    v1="MISSING"
    if csv=$(existing_csv "$q" "$d"); then
      v1=$(sed -n '2p' "$csv" | cut -d, -f2-4)
    fi
    v2_csv="${results_base}/${v2_prefix}_q${q}_delta${d}/results.csv"
    v2="-"
    [[ -f "$v2_csv" ]] && v2=$(sed -n '2p' "$v2_csv" | cut -d, -f2-4)
    printf '%-18s %-30s %-30s\n' "q${q}_delta${d}" "$v1" "$v2"
  done
done
