#!/usr/bin/env bash
# Visualize fitted TrueSpatialLowRankGaussian vs real normal features
# at one (class, patch) using t-SNE.
#
# Usage:
#   ./run_tsne_gaussian_vs_normal.sh                  # uses defaults below
#   ./run_tsne_gaussian_vs_normal.sh screw 648        # classname patch_idx
#   CLASSNAME=pill PATCH_IDX=666 ./run_tsne_gaussian_vs_normal.sh

datapath=/home/maometus/Documents/datasets/mvtec_anomaly_detection

classname=${1:-${CLASSNAME:-screw}}
patch_idx=${2:-${PATCH_IDX:-648}}
num_gaussian_samples=${NUM_GAUSSIAN_SAMPLES:-1000}
num_boundary_samples=${NUM_BOUNDARY_SAMPLES:-0}
boundary_delta=${BOUNDARY_DELTA:-1.0}

python3 tsne_gaussian_vs_normal.py \
--classname "$classname" \
--patch_idx "$patch_idx" \
--data_path "$datapath" \
--gaussian_dir true_spatial_low_rank_gaussian \
--backbone_name wideresnet50 \
-le layer2 \
-le layer3 \
--pretrain_dim 1536 \
--target_dim 1536 \
--patchsize 3 \
--embedding_size 256 \
--resize 329 \
--imagesize_int 288 \
--batch_size 8 \
--num_workers 2 \
--num_gaussian_samples "$num_gaussian_samples" \
--num_boundary_samples "$num_boundary_samples" \
--boundary_delta "$boundary_delta" \
--seed 0 \
--gpu 0
