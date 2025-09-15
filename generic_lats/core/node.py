"""
Node class for tree search algorithms in Generic LATS framework.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import uuid
import time

logger = logging.getLogger(__name__)


class Node:
    """
    Node class for tree search algorithms (LATS, ToT, RAP).
    
    Each node represents a state in the search tree, containing the current
    state information, action that led to it, and metrics for evaluation.
    """
    
    def __init__(self, 
                 state: Optional[Dict[str, Any]] = None,
                 action: str = "",
                 observation: str = "",
                 parent: Optional['Node'] = None,
                 question: str = "",
                 **kwargs):
        """
        Initialize a search tree node.
        
        Args:
            state: Dictionary containing the current state information
            action: Action that led to this state
            observation: Observation received after taking the action
            parent: Parent node in the search tree
            question: The original question/task
            **kwargs: Additional node properties
        """
        # Core node properties
        self.id = str(uuid.uuid4())
        self.state = state or {'thought': '', 'action': action, 'observation': observation}
        self.action = action
        self.observation = observation
        self.parent = parent
        self.question = question
        
        # Tree structure
        self.children: List['Node'] = []
        
        # Search metrics
        self.visits = 0
        self.value = 0.0  # Estimated value of this state
        self.reward = 0.0  # Immediate reward
        self.em = 0  # Exact match or other evaluation metric
        
        # Tree depth and status
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False  # Whether this is a terminal state
        self.is_expanded = False  # Whether children have been generated
        self.exhausted = False   # Whether all children are terminal/evaluated
        
        # Timestamps and metadata
        self.created_at = time.time()
        self.metadata = kwargs
        
        # Add to parent's children if parent exists
        if parent:
            parent.add_child(self)
        
        logger.debug(f"Created node {self.id[:8]} at depth {self.depth}")
    
    def add_child(self, child: 'Node') -> None:
        """
        Add a child node.
        
        Args:
            child: Child node to add
        """
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            child.depth = self.depth + 1
    
    def remove_child(self, child: 'Node') -> None:
        """
        Remove a child node.
        
        Args:
            child: Child node to remove
        """
        if child in self.children:
            self.children.remove(child)
            child.parent = None
    
    def get_trajectory(self) -> List[Dict[str, str]]:
        """
        Get the trajectory from root to this node.
        
        Returns:
            List of dictionaries with action/observation pairs
        """
        trajectory = []
        current = self
        
        # Walk up to root, collecting states
        path = []
        while current is not None:
            path.append(current)
            current = current.parent
        
        # Build trajectory from root to current node
        path.reverse()
        for node in path:
            if node.action or node.observation:  # Skip root if it's empty
                trajectory.append({
                    'action': node.action,
                    'observation': node.observation,
                    'thought': node.state.get('thought', ''),
                    'depth': node.depth
                })
        
        return trajectory
    
    def get_trajectory_string(self) -> str:
        """
        Get trajectory as a formatted string.
        
        Returns:
            Human-readable trajectory string
        """
        trajectory = self.get_trajectory()
        parts = []
        
        for i, step in enumerate(trajectory, 1):
            if step['thought']:
                parts.append(f"Thought {i}: {step['thought']}")
            if step['action']:
                parts.append(f"Action {i}: {step['action']}")
            if step['observation']:
                parts.append(f"Observation {i}: {step['observation']}")
        
        return '\n'.join(parts)
    
    def get_path_to_root(self) -> List['Node']:
        """
        Get list of nodes from this node to root.
        
        Returns:
            List of nodes from current to root
        """
        path = []
        current = self
        
        while current is not None:
            path.append(current)
            current = current.parent
        
        return path
    
    def get_root(self) -> 'Node':
        """
        Get the root node of this tree.
        
        Returns:
            Root node
        """
        current = self
        while current.parent is not None:
            current = current.parent
        return current
    
    def is_leaf(self) -> bool:
        """
        Check if this is a leaf node.
        
        Returns:
            True if node has no children
        """
        return len(self.children) == 0
    
    def is_root(self) -> bool:
        """
        Check if this is the root node.
        
        Returns:
            True if node has no parent
        """
        return self.parent is None
    
    def get_siblings(self) -> List['Node']:
        """
        Get sibling nodes (children of same parent).
        
        Returns:
            List of sibling nodes
        """
        if self.parent is None:
            return []
        return [child for child in self.parent.children if child != self]
    
    def get_descendants(self) -> List['Node']:
        """
        Get all descendant nodes (depth-first).
        
        Returns:
            List of all descendant nodes
        """
        descendants = []
        
        def collect_descendants(node):
            for child in node.children:
                descendants.append(child)
                collect_descendants(child)
        
        collect_descendants(self)
        return descendants
    
    def get_best_child(self, criterion: str = 'value') -> Optional['Node']:
        """
        Get the best child according to some criterion.
        
        Args:
            criterion: Criterion to use ('value', 'visits', 'reward', 'ucb')
            
        Returns:
            Best child node or None if no children
        """
        if not self.children:
            return None
        
        if criterion == 'value':
            return max(self.children, key=lambda x: x.value)
        elif criterion == 'visits':
            return max(self.children, key=lambda x: x.visits)
        elif criterion == 'reward':
            return max(self.children, key=lambda x: x.reward)
        elif criterion == 'ucb':
            # Upper Confidence Bound
            import math
            c = 1.414  # Exploration constant
            
            def ucb_score(node):
                if node.visits == 0:
                    return float('inf')
                exploitation = node.value / node.visits
                exploration = c * math.sqrt(math.log(self.visits) / node.visits)
                return exploitation + exploration
            
            return max(self.children, key=ucb_score)
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
    
    def update_value(self, value: float, visits: int = 1) -> None:
        """
        Update node value and visit count.
        
        Args:
            value: New value to incorporate
            visits: Number of visits to add
        """
        old_total = self.value * self.visits
        self.visits += visits
        new_total = old_total + (value * visits)
        self.value = new_total / self.visits if self.visits > 0 else 0.0
        
        logger.debug(f"Updated node {self.id[:8]}: value={self.value:.3f}, visits={self.visits}")
    
    def backpropagate(self, value: float, visits: int = 1) -> None:
        """
        Backpropagate value update to ancestors.
        
        Args:
            value: Value to propagate
            visits: Number of visits to add
        """
        current = self
        while current is not None:
            current.update_value(value, visits)
            current = current.parent
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert node to dictionary representation.
        
        Returns:
            Dictionary representation of the node
        """
        return {
            'id': self.id,
            'state': self.state,
            'action': self.action,
            'observation': self.observation,
            'question': self.question,
            'depth': self.depth,
            'visits': self.visits,
            'value': self.value,
            'reward': self.reward,
            'em': self.em,
            'is_terminal': self.is_terminal,
            'is_expanded': self.is_expanded,
            'exhausted': self.exhausted,
            'children_count': len(self.children),
            'created_at': self.created_at,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional['Node'] = None) -> 'Node':
        """
        Create node from dictionary representation.
        
        Args:
            data: Dictionary containing node data
            parent: Optional parent node
            
        Returns:
            Node instance
        """
        node = cls(
            state=data.get('state'),
            action=data.get('action', ''),
            observation=data.get('observation', ''),
            parent=parent,
            question=data.get('question', ''),
            **data.get('metadata', {})
        )
        
        # Restore properties
        node.id = data.get('id', node.id)
        node.depth = data.get('depth', node.depth)
        node.visits = data.get('visits', 0)
        node.value = data.get('value', 0.0)
        node.reward = data.get('reward', 0.0)
        node.em = data.get('em', 0)
        node.is_terminal = data.get('is_terminal', False)
        node.is_expanded = data.get('is_expanded', False)
        node.exhausted = data.get('exhausted', False)
        node.created_at = data.get('created_at', node.created_at)
        
        return node
    
    def __repr__(self) -> str:
        """String representation of the node."""
        return (f"Node(id={self.id[:8]}, depth={self.depth}, "
                f"value={self.value:.3f}, visits={self.visits}, "
                f"children={len(self.children)})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        action_str = f"Action: {self.action}" if self.action else "No action"
        obs_str = f"Obs: {self.observation[:50]}..." if len(self.observation) > 50 else f"Obs: {self.observation}"
        return f"{action_str} -> {obs_str} (value={self.value:.3f})"


def node_trajectory_to_text(trajectory: List[Dict[str, str]]) -> str:
    """
    Convert trajectory to text format.
    
    Args:
        trajectory: List of trajectory steps
        
    Returns:
        Formatted trajectory string
    """
    parts = []
    
    for i, step in enumerate(trajectory, 1):
        if step.get('thought'):
            parts.append(f"Thought {i}: {step['thought']}")
        if step.get('action'):
            parts.append(f"Action {i}: {step['action']}")
        if step.get('observation'):
            parts.append(f"Observation {i}: {step['observation']}")
    
    return '\n'.join(parts)