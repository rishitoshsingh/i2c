source activate i2c
which pip

python test.py \
    --data_dir "/scratch/rksing18/datasets/Flickr8" \
    --output_directory "/scratch/rksing18/i2c/Model-Runs/" \
    --experiment_name "30k-20" \
    --encoder "facebook/dinov2-small" \
    --decoder "openai-community/gpt2" \
    --device "gpu" \
    --dataloader_num_workers 1 \