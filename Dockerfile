FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install r1_vlm and VQASynth directly from GitHub
RUN pip3 install git+https://github.com/groundlight/r1_vlm.git
RUN pip3 install git+https://github.com/remyxai/VQASynth.git

# Install additional dependencies
RUN pip3 install depth-anything-v2 segment-anything-2 wandb

# Copy the application code
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh

# Create cache directories for datasets and downloads
RUN mkdir -p .cache/downloaded_images .cache/huggingface/datasets

# Default command - will be overridden by docker run command
CMD ["python3", "demo.py", "--model_path", "models/grpo_ckpt", "--port", "7860", "--host", "0.0.0.0"]