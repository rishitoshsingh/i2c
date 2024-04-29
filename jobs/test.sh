source activate i2c
which pip

python test_trained.py \
    --data_dir "/scratch/rksing18/datasets/Flickr8" \
    --output_directory "/scratch/rksing18/i2c/Model-Runs/" \
    --experiment_name "nlpconnect" \
    --dataloader_num_workers 1 \