#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=in21k_yfcc14m_cc15m_pseudo_small_datasets2_fullset_lp
#SBATCH --partition=learnai4rl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH  --account all

#SBATCH --error=log_small_datasets2_fullset_lp.e.log
#SBATCH --output=log_small_datasets2_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_vit_distill
MODE=linear_probe
CKPT=/checkpoints/sunxm/output/EfficientVLPTaskNew/image_text_unpaired_v2/in21k-cc15m-yfcc14m-512-pseudo_0.3-lr0.0008-wd0.05-bs12288-clip0.0-g24/clip_ViT-B32-in21k-cc15m-yfcc14m-512-pseudo_0.3.yaml_conf~/run_1/324800/default/model_state_dict.pt

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
