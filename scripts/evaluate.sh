#!/bin/bash
set -e

# Evaluation script for CV-Bench
echo "Starting evaluation on CV-Bench..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="spatial_vlm"
export WANDB_NAME="eval_run"

# Set model path (use GRPO checkpoint if available, otherwise SFT)
MODEL_PATH="models/grpo_ckpt"
if [ ! -d "$MODEL_PATH" ]; then
    MODEL_PATH="models/sft_ckpt"
    echo "GRPO checkpoint not found, using SFT checkpoint"
fi

# Run evaluation
python -m src.eval.evaluate \
    --model_path $MODEL_PATH \
    --output reports/results.json \
    --detailed_output reports/detailed_results.json

echo "Evaluation completed! Results saved to reports/results.json"