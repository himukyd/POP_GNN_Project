#!/bin/bash
#SBATCH --job-name=ogbn_run
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=output/himanshu_%j.out
#SBATCH --error=output/himanshu_%j.err

module purge
module load cuda-12.1

source /data/himanshu/env_him/bin/activate

#export CUDA_HOME=/apps/cuda/cuda-12.1
#export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

# 🔍 DEBUG PRINTS
#echo "CUDA_HOME=$CUDA_HOME"
#echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
#ls $CUDA_HOME/targets/x86_64-linux/lib | grep nvrtc

# run
(python himanshu_file.py --dataset reddit && python node_classification.py --dataset reddit --mode puregpu --epoch 50 --num_layers 2 --fan_out 10,10,10 --batch_size 1024) > ./output/reddit.txt