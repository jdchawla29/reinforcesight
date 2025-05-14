# src/eval/evaluate.py
import json
import torch
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import re

try:
    # Try to import HuggingFace datasets
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

try:
    # Try to import specific model implementation for better performance
    from transformers import Qwen2VLForConditionalGeneration
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False

# Default imports that should always work
from transformers import AutoModelForCausalLM, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CVBenchEvaluator:
    """Evaluator for the CV-Bench spatial reasoning benchmark."""
    
    def __init__(self, 
                model_path: str, 
                device: Optional[str] = None,
                config_path: Optional[str] = "configs/base_config.yaml"):
        # Load configuration if provided
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Set device
        self.device = device or self.config.get('system', {}).get('device', 'cuda') 
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Load model and processor
        logger.info(f"Loading model from {model_path}")
        self.model, self.processor = self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load model and processor."""
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        model.eval()
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        return model, processor
    
    def _load_image(self, image_path, image_root=""):
        """Load image with fallbacks."""
        try:
            # Try common paths to find the image
            paths_to_try = [
                image_path,  # Original path
                os.path.join(image_root, image_path) if image_root else None,  # Relative to image_root
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), image_path),  # Project root-relative
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", os.path.basename(image_path)),  # Check datasets directory
                os.path.abspath(image_path)  # Absolute path
            ]

            # Remove None entries if image_root wasn't provided
            paths_to_try = [p for p in paths_to_try if p]

            for path in paths_to_try:
                if os.path.exists(path):
                    logger.info(f"Found image at: {path}")
                    return Image.open(path).convert('RGB')

            # If we can't find the image, log a more detailed error and generate a dummy image
            logger.warning(f"Image not found after trying multiple paths: {image_path}")
            logger.warning(f"Tried paths: {paths_to_try}")
            logger.warning("Using dummy image - this may affect evaluation results")
            return Image.new('RGB', (224, 224), color=(100, 100, 100))

        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {str(e)}")
            return Image.new('RGB', (224, 224), color=(100, 100, 100))
    
    def generate_prediction(self, question: str, image, max_new_tokens: int = 512):
        """Generate prediction for a single question."""
        # Modify question to include reasoning prompt if enabled
        if self.use_reasoning_prompt:
            question = question + "\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags."

        # Create input for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Decode output
        output_text = self.processor.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return output_text

    def generate_batch_predictions(self, questions, images, max_new_tokens: int = 512):
        """Generate predictions for a batch of questions and images."""
        # Modify questions to include reasoning prompt if enabled
        if self.use_reasoning_prompt:
            questions = [q + "\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags." for q in questions]

        # Create messages for each question-image pair
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ] for question in questions
        ]

        # Apply chat template to each message
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        # Process inputs
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )

        # Extract generated text by removing input tokens
        generated_texts = []
        for i, (input_ids, output_ids) in enumerate(zip(inputs.input_ids, outputs)):
            # Get the length of input tokens
            input_length = input_ids.shape[0]
            # Decode only the generated part
            generated_text = self.processor.decode(
                output_ids[input_length:],
                skip_special_tokens=True
            )
            generated_texts.append(generated_text)

        return generated_texts
    
    def extract_answer(self, text: str) -> str:
        """Extract answer from model output using multiple patterns."""
        # First, check for <answer> tags if reasoning prompt was used
        answer_tag_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        if answer_tag_match:
            return answer_tag_match.group(1).strip()

        # Normalize text for consistent processing
        text_lower = text.lower()

        # Define patterns to extract answers in order of preference
        answer_patterns = [
            # Check for explicit "the answer is" pattern with variations
            r"(?:the answer is|my answer is|i conclude that|the answer would be|the correct answer is)[:\s]+([^\.]+)",
            # Check for "therefore" pattern
            r"(?:therefore|thus|hence|so)[,\s]+(?:the answer is|the|in conclusion|we can conclude that|we can say that)?[:\s]*([^\.]+)",
            # Check for pattern with conclusion
            r"(?:in conclusion|to conclude|concluding|as a result)[,\s]+(?:the answer is|the|we can state that|it is clear that|we can see that)?[:\s]*([^\.]+)",
            # Check for spatial relation directly
            r"(?:the|a|is|are)\s+(left|right|above|below|in front of|behind|closer|farther)(?:\s+of|to)?",
            # Final pattern: try to extract the very last part as a failsafe
            r"([^\.]+)$"
        ]

        # Try each pattern in order
        for pattern in answer_patterns:
            matches = re.search(pattern, text_lower)
            if matches:
                answer = matches.group(1).strip()
                if answer:
                    # Clean up the answer
                    answer = re.sub(r'^(is|are|would be|must be|appears to be|seems to be)\s+', '', answer)
                    answer = re.sub(r'[\.,:;!?"\'\)]+$', '', answer)

                    # Check if the answer contains a spatial relation keyword
                    spatial_keywords = ['left', 'right', 'above', 'below', 'in front of', 'behind', 'closer', 'farther']

                    # If we found a good answer with spatial keywords, return it
                    for keyword in spatial_keywords:
                        if keyword in answer:
                            return keyword  # Return just the spatial keyword

                    # If no specific spatial keyword, just return the cleaned answer
                    return answer

        # For multiple choice questions, try to extract letter
        mc_match = re.search(r'\(?([A-F])\)?', text)
        if mc_match:
            return mc_match.group(1).upper()

        # Fallback: split by common delimiters and take the last chunk
        chunks = re.split(r'[\.,:;]', text_lower)
        if chunks:
            last_chunk = chunks[-1].strip()
            # Try to find a spatial relation in the last chunk
            spatial_keywords = ['left', 'right', 'above', 'below', 'in front of', 'behind', 'closer', 'farther']
            for keyword in spatial_keywords:
                if keyword in last_chunk:
                    return keyword
            return last_chunk

        # Ultimate fallback
        return text_lower.strip()[-30:].strip(".,!?\"' ")
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """Evaluate a single sample from the test set."""
        # Extract question and answer
        question = sample['question']
        ground_truth = sample['answer']

        # Load image
        image_path = sample.get('image_path')
        if not image_path:
            # If there's no image path, use a dummy image
            image = Image.new('RGB', (224, 224), color=(100, 100, 100))
        else:
            image = self._load_image(image_path)

        # Generate prediction
        model_output = self.generate_prediction(question, image)

        # Extract answer from prediction
        prediction = self.extract_answer(model_output)

        # Normalize answer for comparison
        normalized_prediction = self._normalize_answer(prediction)
        normalized_ground_truth = self._normalize_answer(ground_truth)

        # Check if prediction is correct using multiple matching strategies
        is_correct = self._check_answer_match(normalized_prediction, normalized_ground_truth)

        # Return evaluation result
        return {
            'question': question,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'normalized_prediction': normalized_prediction,
            'normalized_ground_truth': normalized_ground_truth,
            'model_output': model_output,
            'is_correct': is_correct
        }

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for consistent comparison."""
        # Convert to lowercase
        normalized = answer.lower()

        # Handle common variations
        normalized = re.sub(r'to the (left|right) of', r'\1', normalized)
        normalized = re.sub(r'(above|below) the', r'\1', normalized)
        normalized = re.sub(r'in front of', 'in front of', normalized)  # Standardize spacing

        # Strip extra words
        normalized = re.sub(r'^(is|are|appears to be|seems to be|it is|they are)\s+', '', normalized)
        normalized = re.sub(r'\s+(of|the|than|to|from)\s+.*$', '', normalized)

        # Strip punctuation and extra whitespace
        normalized = re.sub(r'[^\w\s]', '', normalized).strip()

        return normalized

    def _check_answer_match(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth using multiple strategies."""
        # Direct exact match
        if prediction == ground_truth:
            return True

        # Check containment
        if prediction in ground_truth or ground_truth in prediction:
            return True

        # Check for synonyms
        spatial_synonyms = {
            'left': ['leftward', 'leftside'],
            'right': ['rightward', 'rightside'],
            'above': ['over', 'higher', 'top', 'upper'],
            'below': ['under', 'beneath', 'lower', 'bottom'],
            'in front of': ['front', 'foreground', 'nearer', 'closer to camera'],
            'behind': ['background', 'back', 'farther from camera', 'rear'],
            'closer': ['nearer', 'less distant', 'proximate'],
            'farther': ['further', 'more distant', 'remote']
        }

        # Check if ground truth matches any synonym
        for key, synonyms in spatial_synonyms.items():
            if ground_truth == key:
                if any(syn in prediction for syn in synonyms):
                    return True

            # Check if prediction matches any synonym
            if prediction == key:
                if any(syn in ground_truth for syn in synonyms):
                    return True

        return False
    
    def evaluate_dataset(self,
                         data_path: str,
                         output_path: Optional[str] = None,
                         detailed_output: Optional[str] = None) -> Dict:
        """Evaluate the model on a dataset with efficient batching."""
        # Load dataset
        logger.info(f"Loading dataset from {data_path}")
        with open(data_path, 'r') as f:
            dataset = [json.loads(line) for line in f]

        logger.info(f"Evaluating {len(dataset)} samples")

        # Prepare data for batch processing
        all_results = []

        # Process in batches
        for batch_start in tqdm(range(0, len(dataset), self.batch_size), desc="Processing batches"):
            batch_end = min(batch_start + self.batch_size, len(dataset))
            batch = dataset[batch_start:batch_end]

            # Extract questions and images
            questions = [sample['question'] for sample in batch]
            images = []

            for sample in batch:
                image_path = sample.get('image_path')
                if not image_path:
                    # If there's no image path, use a dummy image
                    images.append(Image.new('RGB', (224, 224), color=(100, 100, 100)))
                else:
                    images.append(self._load_image(image_path))

            # Generate predictions in batch
            batch_outputs = self.generate_batch_predictions(questions, images)

            # Process each sample in the batch
            for sample, output in zip(batch, batch_outputs):
                # Extract answer from model output
                prediction = self.extract_answer(output)

                # Normalize for comparison
                normalized_prediction = self._normalize_answer(prediction)
                normalized_ground_truth = self._normalize_answer(sample['answer'])

                # Check if prediction is correct
                is_correct = self._check_answer_match(normalized_prediction, normalized_ground_truth)

                # Create result entry
                result = {
                    'question': sample['question'],
                    'ground_truth': sample['answer'],
                    'prediction': prediction,
                    'normalized_prediction': normalized_prediction,
                    'normalized_ground_truth': normalized_ground_truth,
                    'model_output': output,
                    'is_correct': is_correct,
                    'task': sample.get('task', 'unknown')
                }

                all_results.append(result)

        # Calculate metrics
        metrics = self._calculate_metrics(all_results, dataset)

        # Save results if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                logger.info(f"Results saved to {output_path}")

        # Save detailed results if path is provided
        if detailed_output:
            with open(detailed_output, 'w') as f:
                json.dump(all_results, f, indent=2)
                logger.info(f"Detailed results saved to {detailed_output}")

        return metrics
    
    def _calculate_metrics(self, results: List[Dict], dataset: List[Dict]) -> Dict:
        """Calculate evaluation metrics."""
        # Overall accuracy
        overall_accuracy = sum(r['is_correct'] for r in results) / len(results)
        
        # Accuracy by task type
        task_types = {}
        for result, sample in zip(results, dataset):
            task = sample.get('task', 'unknown')
            if task not in task_types:
                task_types[task] = {
                    'correct': 0,
                    'total': 0
                }
            
            task_types[task]['total'] += 1
            if result['is_correct']:
                task_types[task]['correct'] += 1
        
        # Calculate accuracy by task
        task_accuracy = {}
        for task, counts in task_types.items():
            task_accuracy[task] = counts['correct'] / counts['total']
        
        # Accuracy by relation
        relation_types = {}
        for result, sample in zip(results, dataset):
            relation = sample.get('answer', 'unknown')
            if relation not in relation_types:
                relation_types[relation] = {
                    'correct': 0,
                    'total': 0
                }
            
            relation_types[relation]['total'] += 1
            if result['is_correct']:
                relation_types[relation]['correct'] += 1
        
        # Calculate accuracy by relation
        relation_accuracy = {}
        for relation, counts in relation_types.items():
            relation_accuracy[relation] = counts['correct'] / counts['total']
        
        # Return metrics
        return {
            'overall_accuracy': overall_accuracy,
            'task_accuracy': task_accuracy,
            'relation_accuracy': relation_accuracy,
            'sample_count': len(results)
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on CV-Bench")
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='datasets/cvbench_qual_test.jsonl', help='Path to test data')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='reports/results.json', help='Path to save results')
    parser.add_argument('--detailed_output', type=str, help='Path to save detailed results')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--use_reasoning_prompt', action='store_true', default=True,
                      help='Use reasoning prompt with <think>/<answer> tags')
    parser.add_argument('--no_reasoning_prompt', action='store_false', dest='use_reasoning_prompt',
                      help='Disable reasoning prompt')

    args = parser.parse_args()

    # Create evaluator
    evaluator = CVBenchEvaluator(
        model_path=args.model_path,
        config_path=args.config,
        batch_size=args.batch_size,
        use_reasoning_prompt=args.use_reasoning_prompt
    )

    # Run evaluation
    metrics = evaluator.evaluate_dataset(
        data_path=args.data_path,
        output_path=args.output,
        detailed_output=args.detailed_output
    )

    # Print summary
    print("\nEvaluation Results:")
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print("\nAccuracy by Task:")
    for task, acc in sorted(metrics['task_accuracy'].items()):
        print(f"  {task}: {acc:.4f}")

    print("\nAccuracy by Relation:")
    for relation, acc in sorted(metrics['relation_accuracy'].items()):
        print(f"  {relation}: {acc:.4f}")


if __name__ == "__main__":
    main()