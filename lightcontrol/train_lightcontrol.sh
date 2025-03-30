DATA_ARGS="--webdataset_base_urls \
        /mnt/data/group/text2img_data/style/*/* \
        --num_workers 2 \
        --batch_size 1 \
        --shard_width 5 \
        --train_split 1.0 \
        --val_split 0.0 \
        --test_split 0.0 \
        --resample_train \
        --load_ckpt_id 0 \
        "

MODEL_ARGS="\
  --gradient_accumulation_steps=8 \
  --max_train_steps=2000000 \
  --learning_rate=1e-05 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --mixed_precision="bf16" \
  --checkpointing_steps=5000 \
  --output_dir="result/lightcontrol_19" \
  --max_grad_norm=1 \
  --checkpoints_total_limit=5 \
  "

export options="\
      $DATA_ARGS\
      $MODEL_ARGS"

export CC=gcc
export CXX=g++

accelerate launch --config_file "accelerate_config_debug.yaml" --machine_rank $RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT --num_machines  $WORLD_SIZE  --num_processes 8 train_lightcontrol.py $options
