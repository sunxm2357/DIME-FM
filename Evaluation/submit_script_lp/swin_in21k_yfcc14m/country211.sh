#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=swin_tiny_country211_fullset_lp
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH  --account all

#SBATCH --error=log_country211_fullset_lp.e.log
#SBATCH --output=log_country211_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_swin_tiny
MODE=linear_probe
CKPT=/fsx/sunxm/models/in21k_yfcc14m.pth

cd /fsx/sunxm/code/Elevater_Toolkit_IC

MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG CKPT=$CKPT DATASET=country211  bash run_multi.sh