# src/grpo/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import os

from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

# Import r1_vlm components
from r1_vlm import GRPO_Trainer, GRPOConfig
from r1_vlm.environments.simple_vision_env import SimpleVisionEnvironment

from src.grpo.reward import SpatialReasoningReward

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialReasoningEnvironment(SimpleVisionEnvironment):
    """Custom environment for spatial reasoning tasks using r1_vlm."""
    
    def __init__(self, reward_fn, data_path, image_root="", device=None):
        super().__init__()
        self.reward_fn = reward_fn
        self.data = self._load_data(data_path)
        self.image_root = image_root
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"

    def _load_data(self, data_path):
        """Load data from JSONL file."""
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                data.append(item)
        return data
    
    def _load_image(self, image_path):
        """Load image with fallbacks."""
        try:
            # Try common paths to find the image
            paths_to_try = [
                image_path,  # Original path
                os.path.join(self.image_root, image_path),  # Relative to image_root
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_path),  # Project root-relative
                os.path.abspath(image_path)  # Absolute path
            ]

            for path in paths_to_try:
                if os.path.exists(path):
                    logger.info(f"Found image at: {path}")
                    return Image.open(path).convert('RGB')

            # If we can't find the image, log a more detailed error and generate a dummy image
            logger.warning(f"Image not found after trying multiple paths: {image_path}")
            logger.warning(f"Tried paths: {paths_to_try}")
            logger.warning("Using dummy image - this may affect model performance")
            return Image.new('RGB', (224, 224), color=(100, 100, 100))

        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {str(e)}")
            return Image.new('RGB', (224, 224), color=(100, 100, 100))
    
    def sample_data(self, batch_size=1, task_type=None):
        """Sample data for training, optionally filtering by task type."""
        if task_type:
            filtered_data = [item for item in self.data if item['task'] == task_type]
            if not filtered_data:
                logger.warning(f"No data found for task type {task_type}")
                filtered_data = self.data
        else:
            filtered_data = self.data
            
        indices = np.random.choice(len(filtered_data), size=batch_size, replace=False)
        return [filtered_data[i] for i in indices]
    
    def create_prompt(self, item):
        """Create prompt for a single item."""
        return item['question']
    
    def process_completion(self, item, completion):
        """Process model completion to extract metrics."""
        reward = self.reward_fn.compute_reward(completion, item)
        metrics = {
            'reward': reward,
            'correctness': 1.0 if self.reward_fn._check_answer_correctness(
                self.reward_fn.extract_answer(completion), 
                item['answer']
            ) else 0.0,
            'cot_quality': self.reward_fn._evaluate_cot_quality(completion, item['task']),
            'consistency': self.reward_fn._check_consistency(
                self.reward_fn.extract_answer(completion), 
                completion
            )
        }
        return metrics
    
    def transform_data_for_model(self, items, model, processor):
        """Transform data into format expected by the model."""
        # Create a batch of samples
        batch = []
        for item in items:
            # Load image
            image_path = item.get('image_path')
            if image_path:
                image = self._load_image(image_path)
            else:
                # Generate a dummy image if path not found
                image = Image.new('RGB', (224, 224), color=(100, 100, 100))
            
            # Create prompt
            prompt = self.create_prompt(item)
            
            # Format message
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Add to batch
            batch.append({
                "text": text,
                "image": image,
                "ground_truth": item
            })
            
        return batch


class GRPOTrainer:
    """GRPO Trainer for Spatial Reasoning VLM."""
    
    def __init__(self, config_path, sft_checkpoint=None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and processor
        self.model, self.processor = self._load_model(sft_checkpoint)
        
        # Initialize reward function
        self.reward_fn = SpatialReasoningReward(self.config)
        
        # Initialize environment
        self.env = self._initialize_environment()
        
        # Set random seed
        torch.manual_seed(self.config.get('system', {}).get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.get('system', {}).get('seed', 42))
        np.random.seed(self.config.get('system', {}).get('seed', 42))
    
    def _load_model(self, sft_checkpoint):
        """Load model from SFT checkpoint or base model."""
        if sft_checkpoint:
            logger.info(f"Loading SFT checkpoint from {sft_checkpoint}")
            model_path = sft_checkpoint
        else:
            logger.info(f"Loading base model {self.config['model']['model_id']}")
            model_path = self.config['model']['model_id']
        
        # Load model and processor
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=self.config['model'].get('trust_remote_code', True)
        )
        
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=self.config['model'].get('trust_remote_code', True)
        )
        
        return model, processor
    
    def _initialize_environment(self):
        """Initialize environment for GRPO training."""
        # Get data paths
        train_data_path = self.config['data']['train_path']
        image_root = self.config['data'].get('image_root', '')
        
        # Create environment
        env = SpatialReasoningEnvironment(
            reward_fn=self.reward_fn,
            data_path=train_data_path,
            image_root=image_root,
            device=self.device
        )
        
        return env
    
    def train(self, output_dir=None):
        """Run GRPO training."""
        # Set output directory
        output_dir = output_dir or self.config['model']['output_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create GRPO config
        grpo_config = GRPOConfig(
            num_rollouts=self.config['grpo'].get('num_rollouts', 8),
            num_candidates=self.config['grpo'].get('num_candidates', 9),
            discount_factor=self.config['grpo'].get('discount_factor', 1.0),
            kl_penalty=self.config['grpo'].get('kl_penalty', 0.12),
            checkpoint_frequency=self.config['grpo'].get('checkpoint_freq', 250)
        )
        
        # Create GRPO trainer
        trainer = GRPO_Trainer(
            env=self.env,
            model=self.model,
            processor=self.processor,
            config=grpo_config,
            learning_rate=self.config['training'].get('learning_rate', 3e-5),
            batch_size=self.config['training'].get('batch_size', 32),
            output_dir=output_dir
        )
        
        # Run training
        logger.info("Starting GRPO training")
        trainer.train(
            num_epochs=self.config['training'].get('num_epochs', 5),
            gradient_accumulation_steps=self.config['training'].get('gradient_accumulation_steps', 2),
            warmup_steps=self.config['training'].get('warmup_steps', 50),
            max_grad_norm=self.config['training'].get('max_grad_norm', 0.5)
        )
        
        logger.info(f"Training completed, model saved to {output_dir}")
        return trainer


def main():
    parser = argparse.ArgumentParser(description="Train GRPO model for spatial reasoning")
    parser.add_argument('--config', type=str, default='configs/grpo_config.yaml', help='Path to config file')
    parser.add_argument('--sft_checkpoint', type=str, help='Path to SFT checkpoint')
    parser.add_argument('--output_dir', type=str, help='Directory to save model')
    parser.add_argument('--deepspeed', type=str, help='DeepSpeed config file')
    parser.add_argument('--no_deepspeed', action='store_true', help='Disable DeepSpeed even if config exists')

    args = parser.parse_args()

    # Set DeepSpeed config if provided and not explicitly disabled
    if args.deepspeed and not args.no_deepspeed:
        if os.path.exists(args.deepspeed):
            os.environ['DEEPSPEED_CONFIG'] = args.deepspeed
            print(f"Using DeepSpeed config from {args.deepspeed}")
        else:
            print(f"Warning: DeepSpeed config file {args.deepspeed} not found. Continuing without DeepSpeed.")
    else:
        # Check if we need to unset any existing DeepSpeed config
        if 'DEEPSPEED_CONFIG' in os.environ and args.no_deepspeed:
            del os.environ['DEEPSPEED_CONFIG']
            print("DeepSpeed disabled as requested")

    # Create trainer and run
    trainer = GRPOTrainer(args.config, args.sft_checkpoint)
    trainer.train(output_dir=args.output_dir)


if __name__ == "__main__":
    main()