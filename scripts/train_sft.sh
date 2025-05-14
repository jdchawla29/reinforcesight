#!/bin/bash
set -e

# SFT training script
echo "Starting Supervised Fine-Tuning (SFT)..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="spatial_vlm"
export WANDB_NAME="sft_run"

# Run SFT trainer
python -m src.sft.trainer \
    --config configs/sft_config.yaml \
    --output_dir models/sft_ckpt

echo "SFT training completed!"