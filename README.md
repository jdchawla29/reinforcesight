# ReinforceSight

This project implements a vision-language model (VLM) specialized for qualitative spatial reasoning tasks using supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO).

## Overview

The model is trained to accurately answer qualitative spatial questions about:
- 2D spatial relationships (left/right, above/below)
- 3D depth ordering (front/behind) 
- 3D relative distance (closer/farther)

The system uses:
- **[VQASynth](https://github.com/remyxai/VQASynth)** for synthetic data generation from LocalizedNarratives dataset
- **[TRL](https://github.com/huggingface/trl)** for supervised fine-tuning with LoRA
- **[r1_vlm](https://github.com/groundlight/r1_vlm)** for GRPO alignment
- **[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)** as the base model
- **[CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench)** for evaluation on qualitative spatial reasoning tasks

## Key Features

- Qualitative chain-of-thought reasoning without numerical estimation
- Balanced dataset generation across 8 spatial relation types
- Efficient training with LoRA and 8-bit quantization
- Direct evaluation on CV-Bench qualitative subset
- Integrates with VQASynth's 3D scene reconstruction pipeline

## Installation

## Requirements

- 24GB of VRAM to run the data generation pipelinr
- 1-2 A100s for training

### Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t spatial-vlm .

# Run data generation
docker run --gpus all -v $(pwd):/app spatial-vlm python -m data_gen.generate_dataset

# Run training
docker run --gpus all -v $(pwd):/app spatial-vlm python -m src.sft.trainer
```

### Local Installation

```bash
# Clone the repository
git clone https://github.com/your-username/spatial-vlm.git
cd spatial-vlm

# Create conda environment
conda create -n spatial-vlm python=3.11
conda activate spatial-vlm

# Install dependencies
pip install -r requirements.txt

# Install r1_vlm and VQASynth
pip install git+https://github.com/groundlight/r1_vlm.git
pip install git+https://github.com/remyxai/VQASynth.git

# Install VQASynth dependencies (depth estimation, etc.)
pip install depth-anything-v2 segment-anything-2
```

## Usage

### 1. Generate Synthetic Dataset from LocalizedNarratives

```bash
# Using the script
./scripts/generate_dataset.sh

# Or using Python directly
python -m data_gen.generate_dataset \
    --config configs/config.yaml \
    --num_samples 10000
```

This uses VQASynth to:
- Download images from the HuggingFace LocalizedNarratives dataset
- Process images to extract depth information and object relationships
- Generate balanced questions across 8 spatial relations
- Create qualitative chain-of-thought reasoning
- Create a dataset of 10,000 VQA pairs

### 2. Supervised Fine-Tuning

```bash
# Using the script
./scripts/train_sft.sh

# Or using Python directly
python -m src.sft.trainer --config configs/sft_config.yaml
```

### 3. GRPO Training

```bash
# Using the script
./scripts/train_grpo.sh

# Or using Python directly
python -m src.grpo.trainer --config configs/grpo_config.yaml
```

### 4. Evaluation on CV-Bench

```bash
# Using the script
./scripts/evaluate.sh

# Or using Python directly
python -m src.eval.evaluate \
    --model_path models/grpo_ckpt \
    --output reports/results.json
```

This evaluates on CV-Bench's qualitative subset:
- Spatial Relationship (650 samples)
- Depth Order (600 samples)
- Relative Distance (600 samples)


## Project Structure

```
.
├── configs/                 # Configuration files
│   ├── base_config.yaml     # Base shared configuration
│   ├── config.yaml          # VQASynth configuration
│   ├── sft_config.yaml      # SFT hyperparameters 
│   ├── grpo_config.yaml     # GRPO configuration
│   └── templates.yaml       # Question templates
├── data_gen/                # VQASynth integration
│   ├── generate_dataset.py  # Dataset creation pipeline
│   └── vqasynth_wrapper.py  # Wrapper for VQASynth
├── datasets/                # Generated datasets
├── models/                  # Model checkpoints
│   ├── sft_ckpt/            # SFT checkpoint
│   └── grpo_ckpt/           # GRPO checkpoint
├── reports/                 # Results and logs
├── scripts/                 # Convenience scripts
├── src/                     # Source code
│   ├── sft/                 # Supervised fine-tuning
│   │   └── trainer.py       # SFT trainer
│   ├── grpo/                # GRPO training
│   │   ├── trainer.py       # GRPO trainer
│   │   └── reward.py        # Reward function
│   └── eval/                # Evaluation
│       └── evaluate.py      # CV-Bench evaluation
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python dependencies
└── README.md
```

## Dataset: LocalizedNarratives

This project uses the [LocalizedNarratives dataset](https://huggingface.co/datasets/HuggingFaceM4/LocalizedNarratives) from HuggingFace to generate synthetic spatial reasoning questions. LocalizedNarratives contains:

- Images with detailed captions describing visual content
- Each caption is connected to the corresponding regions in the image
- A diverse set of real-world scenes with complex spatial relationships

Using VQASynth, we process these images to:
1. Extract depth information using depth estimation models
2. Identify objects and their spatial relationships
3. Generate questions about spatial relationships between objects
4. Create chain-of-thought reasoning explanations

## VQASynth Integration

The project integrates with VQASynth's pipeline:
- **Depth Estimation**: UsesVGGT for metric depth
- **Object Detection**: Uses SAM2 for segmentation
- **Scene Understanding**: Uses Molmo-7B for object captions
- **3D Reconstruction**: RANSAC plane fitting for spatial coordinates

## Chain-of-Thought without Numerical Estimation

Following r1 principles, the CoT reasoning avoids numerical values:

```
# Example CoT for spatial relationship
"Looking at the image, I need to determine the spatial relationship 
between the chair and the table. I can see that the chair is positioned 
further to the left side compared to the table. This visual comparison 
allows me to conclude that the chair is left of the table."
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- VQASynth for the 3D scene reconstruction pipeline
- r1_vlm for GRPO implementation
- CV-Bench for evaluation benchmarks
- HuggingFace for the LocalizedNarratives dataset