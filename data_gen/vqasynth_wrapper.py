# data_gen/vqasynth_wrapper.py
import os
import json
import random
import yaml
from typing import Dict, List, Tuple
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

# Import VQASynth components
from vqasynth.depth import DepthEstimator
from vqasynth.embeddings import EmbeddingGenerator
from vqasynth.localize import Localizer
from vqasynth.scene_fusion import SceneFusion
from vqasynth.r1_reasoning import ReasoningGenerator
from vqasynth.prompts import PromptGenerator
from vqasynth.utils import extract_depth_map


class VQASynthWrapper:
    """Wrapper for VQASynth to generate qualitative spatial reasoning data."""
    
    def __init__(self, config_path: str = "configs/config.yaml", device: str = None):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set device
        self.device = device or self.config['vqasynth']['depth']['device']
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
            
        # Initialize VQASynth components
        self._initialize_components()
        
        # Load templates
        self.templates_path = self.config['vqasynth']['prompting']['templates_file']
        with open(self.templates_path, 'r') as f:
            self.templates = yaml.safe_load(f)
    
    def _initialize_components(self):
        """Initialize VQASynth pipeline components."""
        # Initialize depth estimator
        depth_config = self.config['vqasynth']['depth']
        self.depth_estimator = DepthEstimator(
            model=depth_config['model'],
            device=self.device
        )
        
        # Initialize scene fusion
        self.scene_fusion = SceneFusion(
            max_objects=self.config['vqasynth']['scene_fusion']['max_objects'],
            min_area=self.config['vqasynth']['scene_fusion']['min_object_area']
        )
        
        # Initialize localizer
        self.localizer = Localizer(
            model=self.config['vqasynth']['segmentation']['model'],
            device=self.device
        )
        
        # Initialize embeddings generator
        self.embeddings = EmbeddingGenerator()
        
        # Initialize reasoning generator
        self.reasoning = ReasoningGenerator()
        
        # Initialize prompt generator
        self.prompts = PromptGenerator()
    
    def process_image(self, image_path: str) -> Dict:
        """Process image to extract 3D scene information."""
        try:
            # Load image with error handling
            try:
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")

                image = Image.open(image_path).convert('RGB')
                image_np = np.array(image)
            except (FileNotFoundError, IOError) as e:
                print(f"Error loading image {image_path}: {str(e)}")
                raise RuntimeError(f"Failed to load image: {str(e)}")

            # Extract depth map
            try:
                depth_map = self.depth_estimator.predict(image)
            except Exception as e:
                print(f"Depth estimation failed for {image_path}: {str(e)}")
                raise RuntimeError(f"Depth estimation failed: {str(e)}")

            # Detect objects using localizer
            try:
                objects = self.localizer.detect_objects(image)
                if not objects:
                    print(f"Warning: No objects detected in {image_path}")
            except Exception as e:
                print(f"Object detection failed for {image_path}: {str(e)}")
                raise RuntimeError(f"Object detection failed: {str(e)}")

            # Get object embeddings
            try:
                embeddings = self.embeddings.generate_embeddings(image, objects)
            except Exception as e:
                print(f"Embedding generation failed for {image_path}: {str(e)}")
                raise RuntimeError(f"Embedding generation failed: {str(e)}")

            # Fuse scene information
            try:
                scene = self.scene_fusion.fuse_scene(image, depth_map, objects, embeddings)
            except Exception as e:
                print(f"Scene fusion failed for {image_path}: {str(e)}")
                raise RuntimeError(f"Scene fusion failed: {str(e)}")

            # Add image path to scene data
            scene['image_path'] = image_path
            scene['image_size'] = image.size

            return scene

        except Exception as e:
            print(f"Failed to process image {image_path}: {str(e)}")
            # Re-raise with more context
            raise RuntimeError(f"Image processing pipeline failed for {image_path}: {str(e)}")
    
    def generate_2d_spatial_questions(self, scene: Dict) -> List[Dict]:
        """Generate 2D spatial relationship questions."""
        questions = []
        objects = scene['objects']
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i == j:
                    continue
                
                # Calculate relative position
                x1, y1 = obj1['center_2d']
                x2, y2 = obj2['center_2d']
                
                # Determine relation
                if abs(x1 - x2) > abs(y1 - y2):
                    relation = 'left' if x1 < x2 else 'right'
                else:
                    relation = 'above' if y1 < y2 else 'below'
                
                # Generate question from template
                template = random.choice(self.templates['2d_spatial'])
                question = template.format(obj1=obj1['name'], obj2=obj2['name'])
                
                # Generate chain-of-thought reasoning
                cot = self.reasoning.generate_2d_spatial_reasoning(obj1, obj2, relation)
                
                questions.append({
                    'question': question,
                    'answer': relation,
                    'cot': cot,
                    'task': '2d_spatial',
                    'objects': [obj1['name'], obj2['name']],
                    'image_path': scene['image_path']
                })
        
        return questions
    
    def generate_3d_depth_questions(self, scene: Dict) -> List[Dict]:
        """Generate 3D depth ordering questions."""
        questions = []
        objects = scene['objects']
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i == j:
                    continue
                
                # Check depth difference is significant
                depth_diff = abs(obj1['depth'] - obj2['depth'])
                if depth_diff < 0.3:  # Skip ambiguous cases
                    continue
                
                # Determine relation based on depth
                relation = 'in front of' if obj1['depth'] < obj2['depth'] else 'behind'
                
                # Generate question
                template = random.choice(self.templates['3d_depth'])
                question = template.format(obj1=obj1['name'], obj2=obj2['name'])
                
                # Generate CoT
                cot = self.reasoning.generate_3d_depth_reasoning(obj1, obj2, relation)
                
                questions.append({
                    'question': question,
                    'answer': relation,
                    'cot': cot,
                    'task': '3d_depth',
                    'objects': [obj1['name'], obj2['name']],
                    'image_path': scene['image_path']
                })
        
        return questions
    
    def generate_3d_distance_questions(self, scene: Dict) -> List[Dict]:
        """Generate 3D relative distance questions."""
        questions = []
        objects = scene['objects']
        
        # Need at least 3 objects for relative distance
        if len(objects) < 3:
            return questions
        
        for ref_obj in objects:
            for i, obj1 in enumerate(objects):
                for j, obj2 in enumerate(objects):
                    if i >= j or ref_obj['id'] == obj1['id'] or ref_obj['id'] == obj2['id']:
                        continue
                    
                    # Calculate distances
                    dist1 = self._calculate_3d_distance(ref_obj['position_3d'], obj1['position_3d'])
                    dist2 = self._calculate_3d_distance(ref_obj['position_3d'], obj2['position_3d'])
                    
                    # Skip ambiguous cases
                    if abs(dist1 - dist2) < 0.3:
                        continue
                    
                    # Determine relation
                    relation = 'closer' if dist1 < dist2 else 'farther'
                    
                    # Generate question
                    template = random.choice(self.templates['3d_distance'])
                    question = template.format(
                        ref=ref_obj['name'],
                        obj1=obj1['name'],
                        obj2=obj2['name']
                    )
                    
                    # Generate CoT
                    cot = self.reasoning.generate_3d_distance_reasoning(
                        ref_obj, obj1, obj2, relation
                    )
                    
                    questions.append({
                        'question': question,
                        'answer': relation,
                        'cot': cot,
                        'task': '3d_distance',
                        'objects': [ref_obj['name'], obj1['name'], obj2['name']],
                        'image_path': scene['image_path']
                    })
        
        return questions
    
    def _calculate_3d_distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculate Euclidean distance between two 3D points."""
        return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pos1, pos2)))
    
    def balance_dataset(self, all_data: List[Dict], target_per_relation: int) -> List[Dict]:
        """Balance dataset across relation types."""
        # Group by answer (relation)
        relation_groups = {
            'left': [], 'right': [], 'above': [], 'below': [],
            'in front of': [], 'behind': [], 'closer': [], 'farther': []
        }
        
        for item in all_data:
            rel = item['answer']
            if rel in relation_groups:
                relation_groups[rel].append(item)
        
        # Sample equally from each relation type
        balanced_data = []
        for rel, samples in relation_groups.items():
            if samples:
                num_samples = min(len(samples), target_per_relation)
                balanced_data.extend(random.sample(samples, num_samples))
        
        return balanced_data
    
    def generate_dataset_sample(self, image_path: str) -> List[Dict]:
        """Generate all question types for a single image."""
        # Process image
        scene = self.process_image(image_path)
        
        # Generate questions for each task type
        questions = []
        questions.extend(self.generate_2d_spatial_questions(scene))
        questions.extend(self.generate_3d_depth_questions(scene))
        questions.extend(self.generate_3d_distance_questions(scene))
        
        return questions