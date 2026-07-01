#!/usr/bin/env bash
set -euo pipefail

datapath=${DATAPATH:-/home/maometus/Documents/datasets/mvtec_anomaly_detection}
#datasets=('screw' 'pill' 'capsule' 'carpet' 'grid' 'tile' 'wood' 'zipper' 'cable' 'toothbrush' 'transistor' 'metal_nut' 'bottle' 'hazelnut' 'leather')
if [[ -n "${CLASSNAME:-}" ]]; then
  datasets=("$CLASSNAME")
elif [[ -n "${DATASETS:-}" ]]; then
  read -r -a datasets <<< "$DATASETS"
else
  datasets=('screw')
fi
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))

python3 main.py \
--gpu "${GPU:-0}" \
--seed "${SEED:-0}" \
--log_group "${LOG_GROUP:-simplenet_mvtec}" \
--log_project "${LOG_PROJECT:-MVTecAD_Results}" \
--results_path "${RESULTS_PATH:-results}" \
--run_name "${RUN_NAME:-run}" \
net \
-b "${BACKBONE:-wideresnet50}" \
-le layer2 \
-le layer3 \
--pretrain_embed_dimension 1536 \
--target_embed_dimension 1536 \
--patchsize 3 \
--meta_epochs "${META_EPOCHS:-40}" \
--embedding_size 256 \
--gan_epochs "${GAN_EPOCHS:-4}" \
--noise_std "${NOISE_STD:-0.015}" \
--dsc_hidden 1024 \
--dsc_layers 2 \
--dsc_margin "${DSC_MARGIN:-0.5}" \
--pre_proj "${PRE_PROJ:-1}" \
--mix_noise "${MIX_NOISE:-1}" \
dataset \
--batch_size "${BATCH_SIZE:-8}" \
--resize 329 \
--imagesize 288 "${dataset_flags[@]}" mvtec $datapath
