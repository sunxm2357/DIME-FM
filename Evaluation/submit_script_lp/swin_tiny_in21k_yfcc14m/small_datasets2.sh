#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=swin_tiny_small_datasets2_fullset_lp
#SBATCH --partition=hipri
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH  --account all

#SBATCH --error=log_small_datasets2_fullset_lp.e.log
#SBATCH --output=log_small_datasets2_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_swin
MODE=linear_probe
CKPT=/fsx/sunxm/models/in21k_yfcc14m.pth

cd /fsx/sunxm/code/Elevater_Toolkit_IC

# 9
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=oxford-iiit-pets  bash run_multi.sh
# 10
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=patchcamelyon  bash run_multi.sh
# 11
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=rendered-sst2  bash run_multi.sh
# 12
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=resisc45-clip  bash run_multi.sh
# 13
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=stanfordcar  bash run_multi.sh
# 14
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=voc2007classification  bash run_multi.sh
# 15
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=mnist  bash run_multi.sh
# 16
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=fgvc-aircraft-2013b  bash run_multi.sh
