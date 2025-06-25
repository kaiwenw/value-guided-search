#!/bin/bash

python -u inference_eval.py  \
    --benchmark aime-24 \
    --piref_gpu_util 0.75 \
    --piref_model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --seed 7996 \
    --batch_size 10 \
    --num_blocks 8 \
    --block_size 4096 \
    --temperature 0.6 \
    --classifier_ckpt_path VGS-AI/DeepSeek-VM-1.5B \
    --num_repetitions 16 \
    --output_path inference_outputs.jsonl \
    --attention_impl flash_attention_2 \
    --search_type beam2 \
    --neptune_project cornell-rl/oss-infer-eval