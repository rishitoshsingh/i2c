#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 32           # number of cores 
#SBATCH -t 0-4:00:00   # time in d-hh:mm:ss
#SBATCH -p htc      # partition 
#SBATCH -q public       # QOS
#SBATCH --gres=gpu:a30:1
#SBATCH --job-name=i2c-dino-gpt-30k
#SBATCH --mem=32G
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=rksing18@asu.edu
#SBATCH --output=/home/rksing18/i2c/runs/30k/output_multi.out
#SBATCH --error=/home/rksing18/i2c/runs/30k/error_multi.err

source activate i2c
which pip
cd i2c

python main.py \
    --data_dir "/scratch/rksing18/datasets/Flickr30" \
    --output_directory "/scratch/rksing18/i2c/Model-Runs/" \
    --experiment_name "30k-20" \
    --encoder "facebook/dinov2-small" \
    --decoder "openai-community/gpt2" \
    --device "gpu" \
    --lr 0.003 \
    --lr_scheduler_step_size 10 \
    --lr_scheduler_gamma 0.1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 32 \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --gradient_accumulation_steps 4 \
    --weight_decay 0.01
