#!/bin/bash

################## VICUNA ##################
PROMPT_VERSION=v1
# MODEL_VERSION="vicuna-v1-3-7b"
MODEL_VERSION="vicuna-7b-v1.3"
################## VICUNA ##################

RT_rank_llm=4
RT_rank_vision=6
learning_rate=7e-4
max_len=1024
tune_mm_mlp_adapter=false
tune_lm_header=false

PROJ_NAME="mrt-6-4"

deepspeed --num_gpus=2 MRT/train/train_memMRT.py \
    --deepspeed ./scripts/ds_config.json \
    --model_name_or_path /path_to/vicuna-7b-v1.3 \
    --version $PROMPT_VERSION \
    --data_path /path_to/vision_flan/annotation_191-task_1k.json \
    --image_folder /path_to/vision_flan/images_191task_1k \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /path_to/llava-pretrain-vicuna-7b-v1.3/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $PROJ_NAME \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate $learning_rate \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $max_len \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --RT_rank_llm $RT_rank_llm \
    --RT_rank_vision $RT_rank_vision \
    --tune_mm_mlp_adapter $tune_mm_mlp_adapter \
    --tune_lm_header $tune_lm_header \
    --load_best_model_at_end True \
    --metric_for_best_model "train_loss" \
    --greater_is_better false