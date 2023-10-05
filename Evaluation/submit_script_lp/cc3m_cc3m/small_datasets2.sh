#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=cc3m_small_datasets2_fullset_lp
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH  --account all

#SBATCH --error=log_small_datasets2_fullset_lp.e.log
#SBATCH --output=log_small_datasets2_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_vit_distill
MODE=linear_probe
CKPT=/checkpoints/sunxm/output/EfficientVLPTask/image_text_unpaired_v2/cc3m_cc3m_offline-lr0.0008-wd0.05-bs4096-clip0.0-g8/clip_ViT-B32-cc3m-cc3m-512-offline.yaml_conf~/run_2/152000/default/model_state_dict.pt

cd /fsx/sunxm/code/Elevater_Toolkit_IC

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
