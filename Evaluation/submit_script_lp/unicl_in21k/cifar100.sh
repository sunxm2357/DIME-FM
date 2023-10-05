#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=unicl_cifar100_fullset_lp
#SBATCH --partition=hipri
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH  --account all


#SBATCH --error=log_cifar100_fullset_lp.e.log
#SBATCH --output=log_cifar100_fullset_lp.o.log

OUTPUT_DIR=/checkpoints/sunxm/output/evaluations
MODEL_CFG=clip_swin_tiny_my
MODE=linear_probe
CKPT=/checkpoints/sunxm/output/UniCLTask/image_text_pairs_v2/unicl_imagenet21k_image_only_no_decay-lr0.0008-wd0.05-bs4096-clip0.0-g16/swin_tiny_in21k_img_only_no_decay.yaml_conf~/run_3/best_model/model/default/model_state_dict.pt

cd /fsx/sunxm/code/Elevater_Toolkit_IC

MODE=$MODE OUTPUT_DIR=$OUTPUT_DIR MODEL_CFG=$MODEL_CFG mode=$mode CKPT=$CKPT DATASET=cifar100  bash run_multi.sh