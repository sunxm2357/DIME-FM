#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=country211_fullset_lp
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH  --account all

OUTPUT_DIR=/checkpoints/$USER/output/evaluations
MODEL_CFG=clip_vit_distill
MODE=linear_probe
CKPT=$1

cd ../../

MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG CKPT=$CKPT DATASET=country211  bash run_multi.sh