#!/usr/bin/env bash
set -euo pipefail

export CLASSNAME="${CLASSNAME:-screw}"

#bash experiments/run_00_simplenet_noise.sh
bash experiments/run_01_anchored_threshold.sh
bash experiments/run_02_anchored_fixed_radius.sh
bash experiments/run_03_patch_radius_sweep.sh
bash experiments/run_04_anchor_radius_sweep.sh
bash experiments/run_05_sparse_random_sweep.sh
bash experiments/run_06_sparse_block_sweep.sh
bash experiments/run_07_gradient_refinement_sweep.sh
#bash experiments/run_08_default_preprojected_fake.sh
