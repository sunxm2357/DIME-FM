#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=swin_tiny_cifar100_fullset_lp
#SBATCH --partition=hipri
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH  --account all


#SBATCH --error=log_cifar100_fullset_lp.e.log
#SBATCH --output=log_cifar100_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_swin
MODE=linear_probe
CKPT=/fsx/sunxm/models/in21k_yfcc14m.pth

cd /fsx/sunxm/code/Elevater_Toolkit_IC

MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=cifar100  bash run_multi.sh