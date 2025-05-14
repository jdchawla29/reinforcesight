#!/bin/bash
set -e

# GRPO training script
echo "Starting Group Relative Policy Optimization (GRPO) training..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="spatial_vlm"
export WANDB_NAME="grpo_run"

# Check if DeepSpeed should be used
DEEPSPEED_CONFIG="r1_vlm/src/r1_vlm/deepspeed_configs/multi_gpu_3only.yaml"
DEEPSPEED_OPTION=""

if [ -f "$DEEPSPEED_CONFIG" ]; then
    echo "Using DeepSpeed acceleration with config: $DEEPSPEED_CONFIG"
    DEEPSPEED_OPTION="--deepspeed $DEEPSPEED_CONFIG"
else
    echo "DeepSpeed config not found, falling back to standard training"
fi

# Allow disabling DeepSpeed with command line flag
if [ "$1" == "--no-deepspeed" ]; then
    echo "DeepSpeed explicitly disabled via command line argument"
    DEEPSPEED_OPTION="--no_deepspeed"
fi

# Run GRPO trainer
python -m src.grpo.trainer \
    --config configs/grpo_config.yaml \
    --sft_checkpoint models/sft_ckpt \
    --output_dir models/grpo_ckpt \
    $DEEPSPEED_OPTION

echo "GRPO training completed!"