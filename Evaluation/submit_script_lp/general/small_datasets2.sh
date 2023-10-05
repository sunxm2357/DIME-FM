#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=small_datasets2_fullset_lp
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH  --account all

OUTPUT_DIR=/checkpoints/$USER/output/evaluations
MODEL_CFG=clip_vit_distill
MODE=linear_probe
CKPT=$1

cd ../../

# 9
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=oxford-iiit-pets  bash run_multi.sh
# 10
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=patchcamelyon  bash run_multi.sh
# 11
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=rendered-sst2  bash run_multi.sh
# 12
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=resisc45-clip  bash run_multi.sh
# 13
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=stanfordcar  bash run_multi.sh
# 14
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=voc2007classification  bash run_multi.sh
# 15
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=mnist  bash run_multi.sh
# 16
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG  CKPT=$CKPT DATASET=fgvc-aircraft-2013b  bash run_multi.sh
