#!/bin/bash
set -e

# Dataset generation script using LocalizedNarratives
echo "Starting dataset generation with VQASynth and LocalizedNarratives dataset..."

# Create output directories
mkdir -p datasets .cache/downloaded_images

# Run dataset generator
python -m data_gen.generate_dataset \
    --config configs/config.yaml \
    --num_samples 10000

echo "Dataset generation completed! Files saved to datasets/"