#!/usr/bin/env bash
set -euo pipefail

datapath=${DATAPATH:-/home/maometus/Documents/datasets/mvtec_anomaly_detection}
if [[ -n "${CLASSNAME:-}" ]]; then
  datasets=("$CLASSNAME")
elif [[ -n "${DATASETS:-}" ]]; then
  read -r -a datasets <<< "$DATASETS"
else
  datasets=('carpet')
fi
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

neighborhood=${NEIGHBORHOOD:-3}
k=${TSLRG_K:-512}
quantile=${TSLRG_QUANTILE:-0.99}
checkpoint_dir=${TSLRG_CHECKPOINT_DIR:-true_spatial_low_rank_gaussian/runtime_quantiles_v2_k${k}}

args=(
low_rank_gaussian.py
--gpu 0
--seed 0
--log_group simplenet_mvtec
--log_project MVTecAD_Results
--results_path results
--run_name run
--neighborhood "$neighborhood"
--k "$k"
--quantile "$quantile"
--checkpoint_dir "$checkpoint_dir"
)
if [[ "${TSLRG_OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi
args+=(
net
-b wideresnet50
-le layer2
-le layer3
--pretrain_embed_dimension 1536
--target_embed_dimension 1536
--patchsize 3
--meta_epochs 40
--embedding_size 256
--gan_epochs 4
--noise_std 0.015
--dsc_hidden 1024
--dsc_layers 2
--dsc_margin .5
--pre_proj 1
dataset
--batch_size 8
--resize 329
--imagesize 288
"${dataset_flags[@]}"
mvtec "$datapath"
)

python3 "${args[@]}"
