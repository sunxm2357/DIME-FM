#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=cc3m_inpair_cifar10_fullset_lp
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH  --account all

#SBATCH --error=log_cifar10_fullset_lp.e.log
#SBATCH --output=log_cifar10_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/$USER/output/evaluations
MODEL_CFG=clip_vit_distill
MODE=linear_probe
CKPT=$1

cd ../../

OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG MODE=$MODE CKPT=$CKPT DATASET=cifar10  bash run_multi.sh