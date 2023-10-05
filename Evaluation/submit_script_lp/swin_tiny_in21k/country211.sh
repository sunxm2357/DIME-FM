#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=swin_tiny_country211_fullset_lp
#SBATCH --partition=hipri
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH  --account all

#SBATCH --error=log_country211_fullset_lp.e.log
#SBATCH --output=log_country211_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_swin_tiny_distill
MODE=linear_probe
CKPT=/checkpoints/sunxm/output/EfficientVLPTask/image_text_unpaired_v2/swin_tiny_in21k_origin_offline_feat_no_decay-lr0.0008-wd0.05-bs4096-clip0.0-g16/unicl_swin_tiny-in21k-distill-256-offline-feat-no-decay.yaml_conf~/run_1/277000/default/model_state_dict.pt

cd /fsx/sunxm/code/Elevater_Toolkit_IC

MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG CKPT=$CKPT DATASET=country211  bash run_multi.sh