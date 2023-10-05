#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=unicl_small_datasets1_fullset_lp
#SBATCH --partition=hipri
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH  --account all

#SBATCH --error=log_small_datasets1_fullset_lp.e.log
#SBATCH --output=log_small_datasets1_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_swin_tiny_my
MODE=linear_probe
CKPT=/checkpoints/sunxm/output/UniCLTask/image_text_pairs_v2/unicl_imagenet21k_image_only_no_decay-lr0.0008-wd0.05-bs4096-clip0.0-g16/swin_tiny_in21k_img_only_no_decay.yaml_conf~/run_3/best_model/model/default/model_state_dict.pt

cd /fsx/sunxm/code/Elevater_Toolkit_IC

# 1
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=caltech101  bash run_multi.sh
# 2
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=dtd  bash run_multi.sh
# 3
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=eurosat-clip  bash run_multi.sh
#4
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=fer2013  bash run_multi.sh
#5
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=flower102  bash run_multi.sh
# 6
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=gtsrb  bash run_multi.sh
# 7
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=hateful-memes  bash run_multi.sh
# 8
MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=kitti-distance  bash run_multi.sh
