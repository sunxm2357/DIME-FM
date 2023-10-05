#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=unicl_food101_fullset_lp
#SBATCH --partition=hipri
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH  --account all

#SBATCH --error=log_food101_fullset_lp.e.log
#SBATCH --output=log_food101_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_swin_tiny_my
MODE=linear_probe
CKPT=/checkpoints/xide/output/UniCLTask/image_text_pairs_v2/unicl_imagenet21k_yfcc14m_image_only-lr0.0008-wd0.05-bs4096-clip0.0-g16/swin_tiny_in21k_yfcc14m_img_only_no_decay.yaml_conf~/run_1/best_model/model/default/model_state_dict.pt

cd /fsx/sunxm/code/Elevater_Toolkit_IC

MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=food101  bash run_multi.sh