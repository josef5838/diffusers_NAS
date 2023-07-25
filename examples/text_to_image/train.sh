#!/bin/bash

export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="lambdalabs/pokemon-blip-captions"
export OUTPUT_DIR="/home/guest/diffusers_NAS/examples/text_to_image/h_checkpoints"

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=50 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" 
  --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR} \
  ##--use_ema \
  ##--push_to_hub