#!/bin/bash

export HF_HUB_ENABLE_HF_TRANSFER=1
torchrun --standalone --nproc_per_node=4 train_classifier.py \
    --eval_every -1 \
    --save_every 486 \
    --num_steps 12170 \
    --max_lr 0.0001 \
    --data_path VGS-AI/OpenR1-VM \
    --data_num_response 56 \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --total_batch_size 64 \
    --micro_batch_size 8 \
    --small_group_size 2 \
    --run_name single_node_run \
    --track \
    --compile \
    --push_to_hub
