# data_gen/generate_dataset.py
import json
import argparse
import yaml
import os
from pathlib import Path
from typing import Dict, List
import random
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from data_gen.vqasynth_wrapper import VQASynthWrapper
from vqasynth.utils import set_seed


class DatasetGenerator:
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.output_dir = Path(self.config['dataset']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up seed for reproducibility
        set_seed(self.config.get('system', {}).get('seed', 42))
        
        # Initialize VQASynth wrapper
        self.wrapper = VQASynthWrapper(config_path)
        
        # Set up cache directory
        self.cache_dir = Path(self.config['dataset']['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_localized_narratives_dataset(self, num_samples: int = 10000):
        """Generate a dataset using the LocalizedNarratives dataset from HuggingFace."""
        print(f"Loading LocalizedNarratives dataset...")
        
        # Load the LocalizedNarratives dataset
        dataset = load_dataset("HuggingFaceM4/LocalizedNarratives", split="train")
        
        print(f"LocalizedNarratives dataset loaded, containing {len(dataset)} samples")
        print("Generating VQA pairs...")
        
        # Shuffle dataset to get a random subset
        dataset = dataset.shuffle(seed=self.config.get('system', {}).get('seed', 42))
        
        # Take a subset of the dataset for processing
        subset_size = min(num_samples * 3, len(dataset))  # Process more samples than needed to account for failures
        subset = dataset.select(range(subset_size))
        
        all_vqa_pairs = []
        failed_count = 0
        
        # Process each sample
        for i, item in enumerate(tqdm(subset)):
            if len(all_vqa_pairs) >= num_samples:
                break
                
            try:
                # Get image path or URL
                image_path = item.get('image_url')
                
                # If no image URL, try to get image path
                if not image_path:
                    if 'image_path' in item:
                        image_path = item['image_path']
                    elif 'image' in item:
                        image_path = item['image']
                    else:
                        # No image path found, skip
                        failed_count += 1
                        continue
                
                # Download the image if it's a URL
                if image_path.startswith(('http://', 'https://')):
                    try:
                        import requests
                        from PIL import Image
                        from io import BytesIO
                        
                        # Create a directory for downloaded images
                        download_dir = self.cache_dir / "downloaded_images"
                        download_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Generate a filename from the URL
                        filename = f"image_{i}.jpg"
                        local_path = download_dir / filename
                        
                        # Download the image if it doesn't already exist
                        if not local_path.exists():
                            response = requests.get(image_path, timeout=10)
                            if response.status_code == 200:
                                img = Image.open(BytesIO(response.content))
                                img.save(local_path)
                                print(f"Downloaded image to {local_path}")
                            else:
                                print(f"Failed to download image: {response.status_code}")
                                failed_count += 1
                                continue
                        
                        # Update the image path to the local file
                        image_path = str(local_path)
                        
                    except Exception as e:
                        print(f"Error downloading image: {str(e)}")
                        failed_count += 1
                        continue
                
                # Create a scene data structure for VQASynth
                scene_data = {
                    'image_path': image_path,
                    'image_id': f"localized_narrative_{i}",
                    'caption': item.get('caption', ''),
                    'narrative': item.get('narrative', '')
                }
                
                # Generate VQA pairs for this image
                vqa_pairs = self._process_narrative_item(scene_data)
                
                if vqa_pairs:
                    all_vqa_pairs.extend(vqa_pairs)
                    
                    # Print progress periodically
                    if (i + 1) % 10 == 0:
                        print(f"Processed {i + 1} images, generated {len(all_vqa_pairs)} VQA pairs")
                        
            except Exception as e:
                print(f"Error processing item {i}: {str(e)}")
                failed_count += 1
                continue
        
        print(f"Generated {len(all_vqa_pairs)} VQA pairs from LocalizedNarratives")
        print(f"Failed to process {failed_count} items")
        
        # Balance the dataset
        balanced_data = self._balance_dataset(all_vqa_pairs, num_samples)
        
        # Split the dataset
        train_size = int(num_samples * 0.8)
        val_size = int(num_samples * 0.1)
        test_size = num_samples - train_size - val_size
        
        # Shuffle the data
        random.shuffle(balanced_data)
        
        # Split into train, val, test
        train_data = balanced_data[:train_size]
        val_data = balanced_data[train_size:train_size + val_size]
        test_data = balanced_data[train_size + val_size:train_size + val_size + test_size]
        
        # Save to disk
        self._save_split(train_data, 'train')
        self._save_split(val_data, 'val')
        self._save_split(test_data, 'test')
        
        # Save dataset statistics
        self._save_statistics(train_data, val_data, test_data)
        
        print(f"Dataset generation completed successfully!")
        print(f"Train: {len(train_data)} samples")
        print(f"Val: {len(val_data)} samples")
        print(f"Test: {len(test_data)} samples")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def _process_narrative_item(self, item: Dict) -> List[Dict]:
        """Process a LocalizedNarratives item to generate VQA pairs."""
        image_path = item.get('image_path', 'unknown_path')
        try:
            # Validate input data
            if not image_path or not os.path.exists(image_path):
                print(f"Invalid image path: {image_path}")
                return []

            # Use VQASynth to process the image
            try:
                scene = self.wrapper.process_image(image_path)
            except Exception as e:
                print(f"VQASynth processing failed for {image_path}: {str(e)}")
                return []

            # Check if scene contains objects
            if not scene.get('objects') or len(scene.get('objects', [])) < 2:
                print(f"Not enough objects detected in {image_path}, found {len(scene.get('objects', []))} objects")
                return []

            # Add metadata from the item
            scene['caption'] = item.get('caption', '')
            scene['narrative'] = item.get('narrative', '')
            scene['image_id'] = item.get('image_id', '')

            # Generate questions for all task types
            questions = []

            try:
                questions_2d = self.wrapper.generate_2d_spatial_questions(scene)
                questions.extend(questions_2d)
                print(f"Generated {len(questions_2d)} 2D spatial questions for {image_path}")
            except Exception as e:
                print(f"Failed to generate 2D spatial questions for {image_path}: {str(e)}")

            try:
                questions_3d_depth = self.wrapper.generate_3d_depth_questions(scene)
                questions.extend(questions_3d_depth)
                print(f"Generated {len(questions_3d_depth)} 3D depth questions for {image_path}")
            except Exception as e:
                print(f"Failed to generate 3D depth questions for {image_path}: {str(e)}")

            try:
                questions_3d_distance = self.wrapper.generate_3d_distance_questions(scene)
                questions.extend(questions_3d_distance)
                print(f"Generated {len(questions_3d_distance)} 3D distance questions for {image_path}")
            except Exception as e:
                print(f"Failed to generate 3D distance questions for {image_path}: {str(e)}")

            # Add metadata to each question
            for q in questions:
                q['image_path'] = image_path
                q['source_dataset'] = 'localized_narratives'
                q['image_id'] = item.get('image_id', '')
                q['caption'] = item.get('caption', '')
                q['narrative'] = item.get('narrative', '')

            if not questions:
                print(f"Warning: No questions generated for {image_path}")

            return questions

        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _balance_dataset(self, data: List[Dict], target_size: int) -> List[Dict]:
        """Balance dataset to have equal number of samples per task and relation."""
        # Group by task and relation
        task_relation_groups = {}
        
        for item in data:
            task = item['task']
            relation = item['answer']
            key = f"{task}_{relation}"
            
            if key not in task_relation_groups:
                task_relation_groups[key] = []
                
            task_relation_groups[key].append(item)
        
        # Calculate target number per group
        num_groups = len(task_relation_groups)
        target_per_group = target_size // num_groups
        
        # Sample from each group
        balanced_data = []
        for group_key, items in task_relation_groups.items():
            if items:
                # Sample up to target_per_group items from this group
                sampled = random.sample(items, min(target_per_group, len(items)))
                balanced_data.extend(sampled)
        
        # If we don't have enough samples, sample more from groups that have extras
        if len(balanced_data) < target_size:
            remaining = target_size - len(balanced_data)
            extras = []
            
            for group_key, items in task_relation_groups.items():
                if len(items) > target_per_group:
                    extras.extend(items[target_per_group:])
            
            if extras:
                random.shuffle(extras)
                balanced_data.extend(extras[:remaining])
        
        return balanced_data[:target_size]
    
    def _save_split(self, data: List[Dict], split: str):
        """Save dataset split to JSONL file."""
        output_file = self.output_dir / f"cvbench_qual_{split}.jsonl"
        
        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    def _save_statistics(self, train: List[Dict], val: List[Dict], test: List[Dict]):
        """Save dataset statistics."""
        stats = {
            'splits': {
                'train': len(train),
                'val': len(val),
                'test': len(test)
            },
            'tasks': {},
            'relations': {},
            'source_datasets': {}
        }
        
        # Calculate task distribution and other statistics
        for split_name, split_data in [('train', train), ('val', val), ('test', test)]:
            task_counts = {}
            relation_counts = {}
            source_counts = {}
            
            for item in split_data:
                task = item['task']
                relation = item['answer']
                source = item.get('source_dataset', 'unknown')
                
                task_counts[task] = task_counts.get(task, 0) + 1
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
            
            stats['tasks'][split_name] = task_counts
            stats['relations'][split_name] = relation_counts
            stats['source_datasets'][split_name] = source_counts
        
        stats_file = self.output_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative spatial reasoning dataset")
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, help='Override output directory from config')
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of VQA pairs to generate')
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.config)
    
    # Override config with command line arguments if provided
    if args.output_dir:
        generator.output_dir = Path(args.output_dir)
        generator.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset from LocalizedNarratives
    generator.generate_localized_narratives_dataset(args.num_samples)


if __name__ == "__main__":
    main()