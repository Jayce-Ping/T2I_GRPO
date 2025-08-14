#!/bin/bash

GPU_NUM=8 # 2,4,8
MODEL_PATH="/raid/data_qianh/jcy/hugging/models/FLUX.1-dev"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir "data/rl_embeddings" \
    --prompt_path "./data/prompts.txt"


torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding.py \
    --model_path $MODEL_PATH \
    --output_dir "data/rl_embeddings_test" \
    --prompt_path "./data/prompts_test.txt"
