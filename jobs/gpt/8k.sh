#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 16           # number of cores 
#SBATCH -t 0-01:30:00   # time in d-hh:mm:ss
#SBATCH -p htc      # partition 
#SBATCH -q public       # QOS
#SBATCH --gres=gpu:a30:1
#SBATCH --job-name=i2c-dino-gpt-8k
#SBATCH --mem=16G
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=rksing18@asu.edu
#SBATCH --output=/home/rksing18/i2c/runs/gpt/8k/output_multi.out
#SBATCH --error=/home/rksing18/i2c/runs/gpt/8k/error_multi.err

source activate i2c
which pip
cd i2c

python main.py \
    --data_dir "/scratch/rksing18/datasets/Flickr8" \
    --output_directory "/scratch/rksing18/i2c/Model-Runs/" \
    --experiment_name "gpt-8k-5-default" \
    --encoder "facebook/dinov2-small" \
    --decoder "openai-community/gpt2" \
    --device "gpu" \
    --lr 0.0003 \
    --lr_scheduler_step_size 10 \
    --lr_scheduler_gamma 0.1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --dataloader_num_workers 16 \
    --num_train_epochs 5 \
    --logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --gradient_accumulation_steps 2 \
    --weight_decay 0.03 \
    --resume
    
