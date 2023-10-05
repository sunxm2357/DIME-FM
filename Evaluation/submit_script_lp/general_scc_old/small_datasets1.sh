#!/bin/bash -l

#$ -N small_datasets1_lp

#$ -m bea

#$ -M sunxm@bu.edu

# Set SCC project
#$ -P ivc-ml

# Request my job to run on Buy-in Compute group hardware my_project has access to
#$ -l buyin

# Request 4 CPUs
#$ -pe omp 3

# Request 2 GPU
#$ -l gpus=1

# Specify the minimum GPU compute capability
#$ -l gpu_c=7.5

#$ -l gpu_memory=12G

#$ -l h_rt=240:00:00

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load miniconda
module load cuda/11.6
module load gcc
conda activate clip
#conda install -c conda-forge opencv

export PROJECT_PATH=/projectnb/ivc-ml
#
cd $PROJECT_PATH/sunxm/code/Elevater_Toolkit_IC

nvidia-smi

OUTPUT_DIR=./output/evaluations
MODEL_CFG=clip_vit_distill
MODE=linear_probe
CKPT=$1


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
