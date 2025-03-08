#!/usr/bin/bash

torchrun --nnodes 1 --nproc_per_node 8 --node_rank 0 --master_addr 127.0.0.1 --master_port 29500 train_and_infer.py