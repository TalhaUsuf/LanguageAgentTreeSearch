"""
HotPotQA task implementation for Generic LATS framework.
"""

import json
import logging
import re
import string
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter

from .base_task import BaseTask

logger = logging.getLogger(__name__)


class HotPotQATask(BaseTask):
    """
    HotPotQA question answering task implementation.
    
    This task loads HotPotQA dataset and implements evaluation methods
    for multi-hop question answering using Wikipedia search and lookup.
    """
    
    def __init__(self, 
                 dataset_path: str,
                 split: str = "dev",
                 **kwargs):
        """
        Initialize HotPotQA task.
        
        Args:
            dataset_path: Path to HotPotQA dataset JSON file
            split: Dataset split to use
            **kwargs: Additional arguments passed to BaseTask
        """
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.data = []
        
        super().__init__(**kwargs)
        
        # Load dataset
        self._load_dataset()
        
        logger.info(f"Loaded HotPotQA {split} dataset with {len(self.data)} questions")
    
    def _load_dataset(self):
        """Load HotPotQA dataset from JSON file."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Convert to internal format
        self.data = []
        for item in raw_data:
            self.data.append({
                'question': item['question'],
                'answer': item['answer'],
                'id': item.get('_id', len(self.data)),
                'level': item.get('level', 'hard'),
                'type': item.get('type', 'comparison')
            })
    
    def initialize_task(self):
        """Initialize task-specific components."""
        # Set default prompts if not loaded from file
        if not self.prompts:
            self.prompts = {
                'cot_prompt': self._get_default_cot_prompt(),
                'value_prompt': self._get_default_value_prompt(),
                'reflection_prompt': self._get_default_reflection_prompt()
            }
    
    def get_input(self, idx: int) -> str:
        """
        Get the question for a specific task instance.
        
        Args:
            idx: Index of the task instance
            
        Returns:
            Question string
        """
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset size {len(self.data)}")
        
        return self.data[idx]['question']
    
    def evaluate_output(self, idx: int, output: str) -> Dict[str, Any]:
        """
        Evaluate the output against ground truth using exact match and F1 score.
        
        Args:
            idx: Index of the task instance
            output: Generated output to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        if idx >= len(self.data):
            return {
                'is_correct': False,
                'score': 0.0,
                'em': 0,
                'f1': 0.0,
                'ground_truth': 'Unknown',
                'prediction': output,
                'error': f'Index {idx} out of range'
            }
        
        ground_truth = self.data[idx]['answer']
        
        # Normalize answers for comparison
        pred_normalized = self._normalize_answer(output)
        gt_normalized = self._normalize_answer(ground_truth)
        
        # Exact match
        em = int(pred_normalized == gt_normalized)
        
        # F1 score
        f1, precision, recall = self._f1_score(pred_normalized, gt_normalized)
        
        return {
            'is_correct': em == 1,
            'score': f1,  # Use F1 as main score
            'em': em,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'ground_truth': ground_truth,
            'prediction': output,
            'question_id': self.data[idx]['id'],
            'question_level': self.data[idx]['level'],
            'question_type': self.data[idx]['type']
        }
    
    def get_task_description(self) -> str:
        """
        Get description of the HotPotQA task.
        
        Returns:
            Task description string
        """
        return """
        You are solving multi-hop question answering tasks using HotPotQA dataset.
        
        Your goal is to answer complex questions that require reasoning across multiple 
        Wikipedia articles. Use the search and lookup tools to gather information, 
        then provide a precise answer.
        
        Available actions:
        - Search[entity]: Search Wikipedia for an entity
        - Lookup[keyword]: Look up keyword in the current page
        - Finish[answer]: Submit your final answer
        
        Think step by step and gather all necessary information before answering.
        """
    
    def _get_default_cot_prompt(self) -> str:
        """Get default chain-of-thought prompt for HotPotQA."""
        return """
Solve a question answering task with interleaving Thought, Action, Observation steps. 
Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
After each observation, provide the next Thought and next Action. Here are some examples:

Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action 1: Search[Colorado orogeny]
Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas. This took place from 1780 to 1650 million years ago (Mya), during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico. The Colorado orogeny was likely part of the larger Yavapai orogeny.
Thought 2: It does not mention the eastern sector of the Colorado orogeny. I need to look up eastern sector.
Action 2: Lookup[eastern sector]
Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
Thought 3: The eastern sector of Colorado orogeny extends into the High Plains, so High Plains is the area. I need to search High Plains and find its elevation range.
Action 3: Search[High Plains]
Observation 3: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).
Thought 4: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action 4: Finish[1,800 to 7,000 ft]

Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.
Action 1: Search[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. 
Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.
Action 3: Finish[Richard Nixon]

{input}
Current progress:
{trajectory}
"""
    
    def _get_default_value_prompt(self) -> str:
        """Get default value evaluation prompt."""
        return """
Evaluate the following state in solving the question.

Question: {question}
Current state: {state}
Previous failed attempts:
{trajectory}

Based on the progress made and the information gathered, rate the likelihood of successfully solving this question from the current state.

Consider:
- How much relevant information has been gathered
- Whether the search is heading in the right direction
- How close the current state is to finding the answer

Rate the likelihood of success from this state (0-10 where 10 is most likely to succeed):
"""
    
    def _get_default_reflection_prompt(self) -> str:
        """Get default reflection prompt."""
        return """
You are an advanced reasoning agent that can improve based on self reflection. You will be given a previous reasoning trial in which you were given access to a Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.

Previous trial:
{trajectory}

Reflection:
"""
    
    @staticmethod
    def _normalize_answer(s: str) -> str:
        """Normalize answer for comparison."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    @staticmethod
    def _f1_score(prediction: str, ground_truth: str) -> tuple:
        """
        Calculate F1 score between prediction and ground truth.
        
        Returns:
            Tuple of (f1, precision, recall)
        """
        normalized_prediction = HotPotQATask._normalize_answer(prediction)
        normalized_ground_truth = HotPotQATask._normalize_answer(ground_truth)
        
        ZERO_METRIC = (0, 0, 0)
        
        # Handle yes/no/noanswer cases
        if (normalized_prediction in ['yes', 'no', 'noanswer'] and 
            normalized_prediction != normalized_ground_truth):
            return ZERO_METRIC
        if (normalized_ground_truth in ['yes', 'no', 'noanswer'] and 
            normalized_prediction != normalized_ground_truth):
            return ZERO_METRIC
        
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            return ZERO_METRIC
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1, precision, recall
    
    def __len__(self) -> int:
        """Get number of questions in the dataset."""
        return len(self.data)
    
    def get_metrics_names(self) -> List[str]:
        """Get list of metrics that this task can evaluate."""
        return ['accuracy', 'em', 'f1', 'precision', 'recall']
    
    def get_question_info(self, idx: int) -> Dict[str, Any]:
        """
        Get detailed information about a question.
        
        Args:
            idx: Question index
            
        Returns:
            Dictionary with question information
        """
        if idx >= len(self.data):
            return {}
        
        return {
            'index': idx,
            'question': self.data[idx]['question'],
            'answer': self.data[idx]['answer'],
            'id': self.data[idx]['id'],
            'level': self.data[idx]['level'],
            'type': self.data[idx]['type']
        }