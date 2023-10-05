#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=small_datasets1_fullset_lp
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH  --account all

OUTPUT_DIR=/checkpoints/$USER/output/evaluations
MODEL_CFG=clip_vit_distill
MODE=linear_probe
CKPT=$1

cd ../../

# 1
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=caltech101  bash run_multi.sh
# 2
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=dtd  bash run_multi.sh
# 3
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=eurosat-clip  bash run_multi.sh
#4
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=fer2013  bash run_multi.sh
#5
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=flower102  bash run_multi.sh
# 6
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=gtsrb  bash run_multi.sh
# 7
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=hateful-memes  bash run_multi.sh
# 8
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=kitti-distance  bash run_multi.sh
