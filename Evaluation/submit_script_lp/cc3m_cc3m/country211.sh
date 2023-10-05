#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=cc3m_country211_fullset_lp
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH  --account all

#SBATCH --error=log_country211_fullset_lp.e.log
#SBATCH --output=log_country211_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_vit_distill
MODE=linear_probe
CKPT=/checkpoints/sunxm/output/EfficientVLPTask/image_text_unpaired_v2/cc3m_cc3m_offline-lr0.0008-wd0.05-bs4096-clip0.0-g8/clip_ViT-B32-cc3m-cc3m-512-offline.yaml_conf~/run_2/152000/default/model_state_dict.pt

cd /fsx/sunxm/code/Elevater_Toolkit_IC

MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG CKPT=$CKPT DATASET=country211  bash run_multi.sh