# src/sft/trainer.py
import os
import torch
import yaml
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpatialReasoningDataset(Dataset):
    """Dataset for spatial reasoning with CoT."""
    
    def __init__(self, data_path: str, image_root: str, processor, max_length: int = 2048):
        self.data = self._load_data(data_path)
        self.processor = processor
        self.max_length = max_length
        self.image_root = image_root
        
    def _load_data(self, data_path: str) -> List[Dict]:
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def _load_image(self, image_path):
        """Load image from path, or return a dummy image if file doesn't exist."""
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
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = item.get('image_path')
        if not image_path:
            # If image path is not in the data, use the image_id to find it
            image_id = item.get('image_id', f"dummy_image_{idx}")
            # In a real implementation, map image_id to path
            image = Image.new('RGB', (224, 224), color=(100, 100, 100))
        else:
            image = self._load_image(image_path)
        
        # Format the conversation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": item['question']}
                ]
            },
            {
                "role": "assistant",
                "content": f"{item.get('cot', '')}\n\nTherefore, the answer is: {item['answer']}"
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(messages, tokenize=False)
        
        # Process inputs
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "pixel_values": inputs.get("pixel_values", inputs.get("images")).squeeze(),
            "labels": inputs["input_ids"].squeeze()  # For SFT, labels = input_ids
        }


class SpatialVLMTrainer:
    """Trainer for spatial reasoning VLM."""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up device
        self.device = torch.device(self.config.get('system', {}).get('device', 'cuda') 
                                  if torch.cuda.is_available() else 'cpu')
        
        # Load model and processor
        self.model, self.processor = self._initialize_model()
        
        # Apply LoRA
        self.model = self._setup_lora(self.model)
        
        # Set seed for reproducibility
        torch.manual_seed(self.config.get('system', {}).get('seed', 42))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.get('system', {}).get('seed', 42))
        
    def _initialize_model(self):
        """Initialize the base model with quantization."""
        model_id = self.config['model']['model_id']
        logger.info(f"Loading model from {model_id}")
        
        # Quantization config for 8-bit
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4"
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config['model'].get('trust_remote_code', True)
        )
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_id, 
            trust_remote_code=self.config['model'].get('trust_remote_code', True)
        )
        
        # Set pad token if missing
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        return model, processor
    
    def _setup_lora(self, model):
        """Setup LoRA for efficient fine-tuning."""
        lora_config = LoraConfig(
            r=self.config.get('lora', {}).get('rank', 16),
            lora_alpha=self.config.get('lora', {}).get('alpha', 32),
            target_modules=self.config.get('lora', {}).get('target_modules', [
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=self.config.get('lora', {}).get('dropout', 0.05),
            bias=self.config.get('lora', {}).get('bias', 'none'),
            task_type=TaskType.CAUSAL_LM
        )
        
        logger.info("Applying LoRA adapter")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def train(self, output_dir: str = None):
        """Run SFT training."""
        # Set output directory
        output_dir = output_dir or self.config['training']['output_dir']
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get data paths
        train_data_path = self.config['data']['train_path']
        val_data_path = self.config['data']['val_path']
        
        # Get image root directory
        image_root = self.config['data'].get('image_root', '')
        
        logger.info(f"Loading training data from {train_data_path}")
        logger.info(f"Loading validation data from {val_data_path}")
        
        # Create datasets
        train_dataset = SpatialReasoningDataset(
            train_data_path, 
            image_root,
            self.processor, 
            max_length=self.config['data'].get('max_length', 2048)
        )
        
        val_dataset = SpatialReasoningDataset(
            val_data_path, 
            image_root,
            self.processor,
            max_length=self.config['data'].get('max_length', 2048)
        )
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['training'].get('num_train_epochs', 3),
            per_device_train_batch_size=self.config['training'].get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=self.config['training'].get('per_device_eval_batch_size', 4),
            gradient_accumulation_steps=self.config['training'].get('gradient_accumulation_steps', 4),
            learning_rate=self.config['training'].get('learning_rate', 1.5e-4),
            warmup_steps=self.config['training'].get('warmup_steps', 100),
            logging_steps=self.config['training'].get('logging_steps', 10),
            eval_steps=self.config['training'].get('eval_steps', 100),
            save_steps=self.config['training'].get('save_steps', 250),
            evaluation_strategy=self.config['training'].get('evaluation_strategy', 'steps'),
            save_strategy=self.config['training'].get('save_strategy', 'steps'),
            load_best_model_at_end=self.config['training'].get('load_best_model_at_end', True),
            metric_for_best_model=self.config['training'].get('metric_for_best_model', 'eval_loss'),
            greater_is_better=self.config['training'].get('greater_is_better', False),
            fp16=self.config['training'].get('fp16', True),
            gradient_checkpointing=self.config['training'].get('gradient_checkpointing', True),
            max_grad_norm=self.config['training'].get('max_grad_norm', 1.0),
            optim=self.config['training'].get('optim', 'adamw_8bit'),
            report_to=self.config.get('logging', {}).get('report_to', ['tensorboard']),
            run_name=self.config.get('logging', {}).get('run_name', 'spatial_vlm_sft')
        )
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.processor.tokenizer,
            peft_config=None  # We already applied PEFT config
        )
        
        logger.info("Starting training")
        trainer.train()
        
        # Save the trained model
        logger.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        
        return trainer


def main():
    parser = argparse.ArgumentParser(description="Train SFT model for spatial reasoning")
    parser.add_argument('--config', type=str, default='configs/sft_config.yaml', help='Path to config file')
    parser.add_argument('--output_dir', type=str, help='Directory to save model')
    
    args = parser.parse_args()
    
    trainer = SpatialVLMTrainer(args.config)
    trainer.train(output_dir=args.output_dir)


if __name__ == "__main__":
    main()