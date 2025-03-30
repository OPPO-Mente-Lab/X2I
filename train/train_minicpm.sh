export NCCL_P2P_LEVEL=NVL


DATA_ARGS="\
        --webdataset_base_urls \
        /mnt/data/group/text2img_data/flux_bench/*/*\
        --num_workers 0 \
        --batch_size 1 \
        --shard_width 5 \
        --train_split 1.0 \
      
        "

MODEL_ARGS="\
  --gradient_accumulation_steps=1 \
  --max_train_steps=100000 \
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=100 \
  --checkpointing_steps=2000 \
  --output_dir="result/MiniCPM" \
  --max_grad_norm=1 \
  --checkpoints_total_limit=5 \
  --use_8bit_adam \
  "
# #   --report_to="tensorboard" \
export options="\
      $DATA_ARGS\
      $MODEL_ARGS"

export CC=gcc
export CXX=g++

# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT  -m train_minicpm $options
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=10014  -m train_minicpm $options
