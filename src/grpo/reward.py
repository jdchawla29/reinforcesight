# src/grpo/reward.py
import torch
import torch.nn.functional as F
from typing import Dict, List, Any, Union
import numpy as np
import re

class SpatialReasoningReward:
    """Custom reward function for spatial reasoning tasks."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.weights = {
            'correct_answer': config.get('reward', {}).get('correct_answer_weight', 1.0),
            'incorrect_answer': config.get('reward', {}).get('incorrect_answer_weight', -0.5),
            'cot_quality': config.get('reward', {}).get('cot_quality_weight', 0.3),
            'consistency': config.get('reward', {}).get('consistency_weight', 0.2)
        }
        
        self.relation_embeddings = self._initialize_relation_embeddings()
    
    def _initialize_relation_embeddings(self):
        """Initialize embeddings for spatial relations."""
        relations = ['left', 'right', 'above', 'below', 
                    'in front of', 'behind', 'closer', 'farther']
        
        # Create simple one-hot embeddings for relations
        embeddings = {}
        for i, rel in enumerate(relations):
            embed = torch.zeros(len(relations))
            embed[i] = 1.0
            embeddings[rel] = embed
        return embeddings
    
    def extract_answer(self, text: str) -> str:
        """Extract the answer from model output."""
        # Try to find the answer after a pattern like "the answer is" or "Therefore,"
        patterns = [
            r"(?:Therefore,|Thus,|In conclusion,|So,)(?:.*)(?:answer is:?|object is:?)(.*)",
            r"(?:The answer is:?|The object is:?)(.*)",
            r"(?:My answer is:?|My response is:?)(.*)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip().lower()
                # Remove punctuation at the end
                answer = re.sub(r'[.!,;:]$', '', answer).strip()
                return answer
        
        # If no pattern matches, return the last sentence as a fallback
        sentences = text.split('.')
        if sentences:
            return sentences[-1].strip().lower()
        
        return text.strip().lower()
    
    def compute_reward(self, 
                     model_output: str, 
                     ground_truth: Dict[str, Any]) -> float:
        """
        Compute reward based on correctness and CoT quality.
        
        Args:
            model_output: Text output from the model
            ground_truth: Ground truth data with keys 'answer', 'cot', 'task'
            
        Returns:
            Reward score
        """
        # Extract predicted answer
        predicted_answer = self.extract_answer(model_output)
        gt_answer = ground_truth['answer'].lower()
        task_type = ground_truth['task']
        
        # Split the model output to separate CoT from answer
        cot = model_output
        
        # Base reward for correct answer
        is_correct = self._check_answer_correctness(predicted_answer, gt_answer)
        reward = self.weights['correct_answer'] if is_correct else self.weights['incorrect_answer']
        
        # CoT quality bonus
        cot_quality = self._evaluate_cot_quality(cot, task_type)
        reward += cot_quality * self.weights['cot_quality']
        
        # Consistency bonus
        consistency = self._check_consistency(predicted_answer, cot)
        reward += consistency * self.weights['consistency']
        
        return reward
    
    def _check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Check if the predicted answer is correct."""
        # Normalize predictions and ground truth
        pred_norm = predicted.strip().lower()
        gt_norm = ground_truth.strip().lower()
        
        # Direct match
        if pred_norm == gt_norm:
            return True
            
        # Check for synonyms and alternative phrasings
        spatial_synonyms = {
            'left': ['to the left of', 'on the left side of', 'leftward of'],
            'right': ['to the right of', 'on the right side of', 'rightward of'],
            'above': ['over', 'on top of', 'higher than', 'up from'],
            'below': ['under', 'beneath', 'lower than', 'down from'],
            'in front of': ['before', 'ahead of', 'closer to camera', 'nearer to viewer'],
            'behind': ['after', 'in back of', 'farther from camera', 'further from viewer'],
            'closer': ['nearer', 'more proximate', 'less distant'],
            'farther': ['further', 'more distant', 'less proximate']
        }
        
        # Check if ground truth is in our synonyms dictionary
        if gt_norm in spatial_synonyms:
            # Check if prediction matches any synonym
            if any(syn in pred_norm for syn in spatial_synonyms[gt_norm]):
                return True
                
        return False
    
    def _evaluate_cot_quality(self, cot: str, task_type: str) -> float:
        """Evaluate quality of chain-of-thought reasoning."""
        score = 0.0
        
        # Check for key reasoning elements based on task type
        if task_type == '2d_spatial':
            keywords = ['position', 'located', 'left', 'right', 'above', 'below',
                        'compare', 'comparing', 'side', 'higher', 'lower']
        elif task_type == '3d_depth':
            keywords = ['depth', 'distance', 'camera', 'closer', 'farther', 'front',
                       'behind', 'occlusion', 'perspective', 'viewer']
        elif task_type == '3d_distance':
            keywords = ['distance', 'relative', 'closer', 'farther', 'proximity',
                       'nearer', 'reference', 'compared to']
        else:
            keywords = []
        
        # Count relevant keywords
        cot_lower = cot.lower()
        keyword_count = sum(1 for kw in keywords if kw in cot_lower)
        score += min(keyword_count / max(len(keywords) * 0.5, 1), 1.0) * 0.4
        
        # Check for structured reasoning
        reasoning_markers = ['therefore', 'thus', 'because', 'since', 'as a result',
                           'this means', 'which indicates', 'this shows', 'conclude']
        if any(marker in cot_lower for marker in reasoning_markers):
            score += 0.3
        
        # Check for visual references
        visual_refs = ['see', 'observe', 'appear', 'visual', 'image', 'looking at',
                     'notice', 'visible', 'position', 'located']
        if any(ref in cot_lower for ref in visual_refs):
            score += 0.2
        
        # Length check (not too short or too long)
        word_count = len(cot.split())
        if 20 <= word_count <= 150:
            score += 0.1
        
        return min(score, 1.0)
    
    def _check_consistency(self, prediction: str, cot: str) -> float:
        """Check if CoT is consistent with prediction."""
        pred_lower = prediction.lower()
        cot_lower = cot.lower()
        
        # Simple check: does the CoT mention the predicted relation?
        if pred_lower in cot_lower:
            return 1.0
            
        # Check if any synonym of the prediction is in the CoT
        spatial_synonyms = {
            'left': ['to the left of', 'on the left side', 'leftward'],
            'right': ['to the right of', 'on the right side', 'rightward'],
            'above': ['over', 'on top of', 'higher than', 'up from'],
            'below': ['under', 'beneath', 'lower than', 'down from'],
            'in front of': ['before', 'ahead of', 'closer to camera', 'nearer to viewer'],
            'behind': ['after', 'in back of', 'farther from camera', 'further from viewer'],
            'closer': ['nearer', 'more proximate', 'less distant'],
            'farther': ['further', 'more distant', 'less proximate']
        }
        
        for rel, synonyms in spatial_synonyms.items():
            if rel in pred_lower:
                if any(syn in cot_lower for syn in synonyms):
                    return 0.8
        
        # Check if reasoning conclusion words are followed by the prediction
        conclusion_markers = ['therefore', 'thus', 'so', 'hence', 'conclude', 'answer']
        for marker in conclusion_markers:
            if marker in cot_lower:
                marker_idx = cot_lower.find(marker)
                conclusion_text = cot_lower[marker_idx:]
                if pred_lower in conclusion_text:
                    return 0.9
        
        return 0.0