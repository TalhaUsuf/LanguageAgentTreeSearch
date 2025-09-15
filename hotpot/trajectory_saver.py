"""
Trajectory saving utilities for LATS experiments.
Saves detailed trajectories in training-ready format.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any
from logger_config import get_context_logger

logger = get_context_logger()


class TrajectorySaver:
    """Handles saving LATS trajectories for training data."""
    
    def __init__(self, trajs_dir: str = "trajs"):
        """
        Initialize trajectory saver.
        
        Args:
            trajs_dir: Directory to save trajectories
        """
        self.trajs_dir = trajs_dir
        self.trajectories = []
        os.makedirs(trajs_dir, exist_ok=True)
        
    def add_trajectory(self, 
                      question: str,
                      trajectory_nodes: List[Any],
                      final_answer: str,
                      reward: float,
                      em: float,
                      iteration: int,
                      question_idx: int):
        """
        Add a trajectory to the collection.
        
        Args:
            question: The original question
            trajectory_nodes: List of nodes in the trajectory
            final_answer: Final answer provided
            reward: Reward received
            em: Exact match score
            iteration: LATS iteration when trajectory was found
            question_idx: Index of the question in dataset
        """
        # Convert trajectory nodes to structured format
        trajectory_steps = []
        for node in trajectory_nodes:
            if hasattr(node, 'state') and node.state:
                step = {
                    "depth": getattr(node, 'depth', 0),
                    "thought": node.state.get('thought', ''),
                    "action": node.state.get('action', ''),
                    "observation": node.state.get('observation', ''),
                    "value": getattr(node, 'value', 0.0),
                    "visits": getattr(node, 'visits', 0),
                    "is_terminal": getattr(node, 'is_terminal', False),
                    "reward": getattr(node, 'reward', 0.0)
                }
                trajectory_steps.append(step)
        
        # Create training-ready trajectory
        trajectory_data = {
            "timestamp": datetime.now().isoformat(),
            "question_idx": question_idx,
            "question": question,
            "trajectory": trajectory_steps,
            "final_answer": final_answer,
            "reward": reward,
            "exact_match": em,
            "iteration_found": iteration,
            "trajectory_length": len(trajectory_steps),
            "success": reward > 0,
            "reasoning_chain": self._extract_reasoning_chain(trajectory_steps)
        }
        
        self.trajectories.append(trajectory_data)
        logger.debug(f"Added trajectory for Q{question_idx}: success={reward > 0}, length={len(trajectory_steps)}")
        
    def _extract_reasoning_chain(self, trajectory_steps: List[Dict]) -> str:
        """
        Extract a clean reasoning chain for training purposes.
        
        Args:
            trajectory_steps: List of trajectory steps
            
        Returns:
            Clean reasoning chain as string
        """
        reasoning_parts = []
        for step in trajectory_steps:
            if step['thought']:
                reasoning_parts.append(f"Thought {step['depth']}: {step['thought']}")
            if step['action']:
                reasoning_parts.append(f"Action {step['depth']}: {step['action']}")
            if step['observation'] and step['depth'] > 0:
                reasoning_parts.append(f"Observation {step['depth']}: {step['observation']}")
        
        return "\n".join(reasoning_parts)
    
    def save_trajectories(self, filename: str = None):
        """
        Save all collected trajectories to file.
        
        Args:
            filename: Optional filename, defaults to timestamped file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lats_trajectories_{timestamp}.json"
        
        filepath = os.path.join(self.trajs_dir, filename)
        
        # Create summary statistics
        total_trajectories = len(self.trajectories)
        successful_trajectories = sum(1 for t in self.trajectories if t['success'])
        avg_length = sum(t['trajectory_length'] for t in self.trajectories) / total_trajectories if total_trajectories > 0 else 0
        
        output_data = {
            "metadata": {
                "total_trajectories": total_trajectories,
                "successful_trajectories": successful_trajectories,
                "success_rate": successful_trajectories / total_trajectories if total_trajectories > 0 else 0.0,
                "average_trajectory_length": avg_length,
                "generation_timestamp": datetime.now().isoformat()
            },
            "trajectories": self.trajectories
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Saved {total_trajectories} trajectories to {filepath}")
        logger.info(f"Success rate: {successful_trajectories}/{total_trajectories} ({successful_trajectories/total_trajectories*100:.1f}%)")
        
        return filepath
    
    def save_training_format(self, filename: str = None):
        """
        Save trajectories in a format optimized for LLM training.
        
        Args:
            filename: Optional filename for training data
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lats_training_data_{timestamp}.jsonl"
        
        filepath = os.path.join(self.trajs_dir, filename)
        
        with open(filepath, 'w') as f:
            for traj in self.trajectories:
                # Create training example
                training_example = {
                    "instruction": f"Answer the following question using step-by-step reasoning: {traj['question']}",
                    "input": "",
                    "output": traj['reasoning_chain'],
                    "success": traj['success'],
                    "reward": traj['reward'],
                    "exact_match": traj['exact_match']
                }
                f.write(json.dumps(training_example) + '\n')
        
        logger.info(f"Saved training data to {filepath}")
        return filepath


# Global trajectory saver instance
trajectory_saver = TrajectorySaver()