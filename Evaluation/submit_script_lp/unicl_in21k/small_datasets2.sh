#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=in1k_b_small_datasets2_fullset_lp
#SBATCH --partition=hipri
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH  --account all

#SBATCH --error=log_small_datasets2_fullset_lp.e.log
#SBATCH --output=log_small_datasets2_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_swin_tiny_my
MODE=linear_probe
CKPT=/checkpoints/sunxm/output/UniCLTask/image_text_pairs_v2/unicl_imagenet21k_image_only_no_decay-lr0.0008-wd0.05-bs4096-clip0.0-g16/swin_tiny_in21k_img_only_no_decay.yaml_conf~/run_3/best_model/model/default/model_state_dict.pt

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
