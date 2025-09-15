"""
Generic LATS (Language Agent Tree Search) algorithm implementation.
"""

import logging
import random
import time
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np

try:
    from .node import Node
    from ..tasks.base_task import BaseTask
    from ..tools.tool_registry import ToolRegistry
    from ..models.llm_wrapper import LangChainLLM
except ImportError:
    from core.node import Node
    from tasks.base_task import BaseTask
    from tools.tool_registry import ToolRegistry
    from models.llm_wrapper import LangChainLLM

logger = logging.getLogger(__name__)


class LATSSearch:
    """
    Language Agent Tree Search (LATS) implementation.
    
    This class implements the LATS algorithm for solving complex reasoning tasks
    through iterative tree search with language model generation and evaluation.
    """
    
    def __init__(self,
                 task: BaseTask,
                 llm: LangChainLLM,
                 tool_registry: ToolRegistry,
                 config: Dict[str, Any]):
        """
        Initialize LATS search.
        
        Args:
            task: Task instance to solve
            llm: Language model wrapper
            tool_registry: Registry of available tools
            config: Algorithm configuration
        """
        self.task = task
        self.llm = llm
        self.tool_registry = tool_registry
        self.config = config
        
        # Algorithm parameters
        self.iterations = config.get('iterations', 30)
        self.n_generate_sample = config.get('n_generate_sample', 5)
        self.n_evaluate_sample = config.get('n_evaluate_sample', 1)
        self.max_depth = config.get('max_depth', 7)
        self.cache_values = config.get('cache_values', True)
        
        # Reflection settings
        self.reflection_config = config.get('reflection', {})
        self.reflection_enabled = self.reflection_config.get('enabled', True)
        self.max_reflections = self.reflection_config.get('max_reflections', 3)
        
        # State tracking
        self.value_cache = {} if self.cache_values else None
        self.failed_trajectories = []
        self.reflection_map = []
        
        logger.info(f"Initialized LATS search with {self.iterations} iterations")
    
    def search(self, 
               question: str, 
               task_index: int,
               iterations: Optional[int] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Perform LATS search to solve a question.
        
        Args:
            question: Question/task to solve
            task_index: Index of the task
            iterations: Number of search iterations (overrides config)
            **kwargs: Additional search parameters
            
        Returns:
            Dictionary containing search results
        """
        start_time = time.time()
        
        # Override iterations if provided
        if iterations is not None:
            search_iterations = iterations
        else:
            search_iterations = self.iterations
        
        logger.info(f"Starting LATS search for question: {question[:100]}...")
        
        # Reset state
        self.failed_trajectories = []
        self.reflection_map = []
        if self.value_cache:
            self.value_cache.clear()
        
        # Initialize root node
        root = Node(
            state={'thought': '', 'action': '', 'observation': ''},
            question=question,
            parent=None
        )
        
        all_nodes = []
        terminal_nodes = []
        
        # Main search loop
        for iteration in range(search_iterations):
            logger.info(f"Iteration {iteration + 1}/{search_iterations}")
            
            # Select node for expansion
            node = self._select_node(root)
            
            if node is None:
                logger.info("All paths exhausted, ending search")
                break
            
            # Check if we found a successful terminal node
            if node.is_terminal and node.reward == 1:
                logger.info(f"Found successful solution at iteration {iteration + 1}")
                break
            
            # Expand the selected node
            self._expand_node(node, question)
            
            # If node became terminal or has no children, reselect
            while node.is_terminal or not node.children:
                logger.debug("Reselecting due to terminal node or no children")
                node = self._select_node(root)
                if node is None:
                    break
                self._expand_node(node, question)
            
            if node is None:
                break
            
            # Evaluate children
            self._evaluate_node(node, question)
            
            # Rollout from best child
            if node.children:
                best_child = max(node.children, key=lambda x: x.value)
                reward, terminal_node = self._rollout(best_child, question, max_depth=4)
                terminal_nodes.append(terminal_node)
                
                # Check if rollout found solution
                if terminal_node.reward == 1:
                    logger.info("Found successful solution during rollout")
                    break
                
                # Backpropagate reward
                self._backpropagate(terminal_node, reward)
            
            # Collect all nodes
            all_nodes = self._collect_all_nodes(root)
            
            # Check for any successful terminal nodes
            successful_nodes = [n for n in all_nodes if n.is_terminal and n.reward == 1]
            if successful_nodes:
                logger.info(f"Found successful terminal node at iteration {iteration + 1}")
                break
        
        # Find best solution
        all_nodes_list = self._collect_all_nodes(root) + terminal_nodes
        best_node = max(all_nodes_list, key=lambda x: x.reward) if all_nodes_list else root
        
        search_time = time.time() - start_time
        
        # Extract final answer
        final_answer = self._extract_answer(best_node)
        
        # Evaluate against ground truth
        evaluation = self.task.evaluate_output(task_index, final_answer)
        
        # Prepare results
        results = {
            'answer': final_answer,
            'ground_truth': evaluation.get('ground_truth', 'Unknown'),
            'is_correct': evaluation.get('is_correct', False),
            'score': evaluation.get('score', 0.0),
            'reward': best_node.reward,
            'em': best_node.em,
            'iterations_used': iteration + 1,
            'search_time': search_time,
            'nodes_explored': len(all_nodes_list),
            'trajectory': best_node.get_trajectory(),
            'search_tree': self._tree_to_dict(root),
            'tokens_used': self.llm.get_usage_stats().get('total_tokens', 0),
            'api_calls': self.llm.get_usage_stats().get('api_calls', 0)
        }
        
        logger.info(f"Search completed: correct={results['is_correct']}, "
                   f"score={results['score']:.3f}, time={search_time:.2f}s")
        
        return results
    
    def _select_node(self, root: Node) -> Optional[Node]:
        """
        Select a node for expansion using UCT (Upper Confidence bound applied to Trees).
        
        Args:
            root: Root node of the search tree
            
        Returns:
            Selected node or None if all paths exhausted
        """
        node = root
        
        while node and node.children:
            logger.debug(f"Selecting from {len(node.children)} children at depth {node.depth}")
            
            # Check if all children are terminal
            terminal_children = [child for child in node.children if child.is_terminal]
            
            if len(terminal_children) == len(node.children):
                logger.debug(f"All children terminal at depth {node.depth}, backtracking")
                if node.parent:
                    node.parent.children.remove(node)
                node = node.parent
                continue
            
            # Check for successful terminal children
            successful_terminal = next(
                (child for child in terminal_children if child.reward == 1), 
                None
            )
            if successful_terminal:
                logger.info(f"Found successful terminal node at depth {node.depth}")
                return successful_terminal
            
            # Select non-terminal child with highest UCT value
            non_terminal_children = [child for child in node.children if not child.is_terminal]
            if non_terminal_children:
                node = max(non_terminal_children, key=lambda x: self._uct_score(x))
                logger.debug(f"Selected node at depth {node.depth} with UCT {self._uct_score(node):.3f}")
            else:
                node = None
                break
        
        return node
    
    def _uct_score(self, node: Node) -> float:
        """
        Calculate Upper Confidence bound applied to Trees (UCT) score.
        
        Args:
            node: Node to calculate UCT score for
            
        Returns:
            UCT score
        """
        if node.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        if node.parent is None or node.parent.visits == 0:
            return node.value / node.visits
        
        exploitation = node.value / node.visits
        exploration = math.sqrt(2 * math.log(node.parent.visits) / node.visits)
        
        return exploitation + exploration
    
    def _expand_node(self, node: Node, question: str) -> None:
        """
        Expand a node by generating new child states.
        
        Args:
            node: Node to expand
            question: Original question/task
        """
        if node.depth >= self.max_depth:
            logger.info(f"Depth limit reached at {node.depth}")
            node.is_terminal = True
            return
        
        # Generate new states
        new_nodes = self._generate_new_states(node, question)
        node.children.extend(new_nodes)
        node.is_expanded = True
        
        logger.debug(f"Expanded node with {len(new_nodes)} children")
    
    def _generate_new_states(self, node: Node, question: str) -> List[Node]:
        """
        Generate new child states from a parent node.
        
        Args:
            node: Parent node
            question: Original question
            
        Returns:
            List of new child nodes
        """
        # Generate current trajectory
        trajectory = node.get_trajectory_string()
        
        # Create prompt for action generation
        prompt = self.task.cot_prompt_wrap(
            question, 
            trajectory, 
            self.reflection_map
        )
        
        # Generate multiple action samples
        try:
            sampled_actions = self.llm.generate(
                prompt, 
                n=self.n_generate_sample,
                stop=['\nObservation:', 'Observation:']
            )
        except Exception as e:
            logger.error(f"Error generating actions: {e}")
            return []
        
        logger.debug(f"Generated {len(sampled_actions)} action samples")
        
        # Process each sampled action
        unique_states = {}
        
        for action_text in sampled_actions:
            try:
                new_node = self._process_action_sample(node, action_text, question)
                if new_node:
                    # Use thought+action as unique key
                    unique_key = f"{new_node.state.get('thought', '')}::{new_node.action}"
                    unique_states[unique_key] = new_node
            except Exception as e:
                logger.error(f"Error processing action sample: {e}")
                continue
        
        return list(unique_states.values())
    
    def _process_action_sample(self, parent: Node, action_text: str, question: str) -> Optional[Node]:
        """
        Process a single action sample and create a new node.
        
        Args:
            parent: Parent node
            action_text: Generated action text
            question: Original question
            
        Returns:
            New node or None if processing failed
        """
        lines = action_text.strip().split('\n')
        
        thought = ""
        action = ""
        
        # Parse thought and action from generated text
        for line in lines:
            line = line.strip()
            if line.startswith(f"Thought {parent.depth + 1}:"):
                thought = line.split(":", 1)[1].strip()
            elif line.startswith("Action") and ":" in line:
                action = line.split(":", 1)[1].strip()
        
        if not action:
            logger.debug("No action found in sample")
            return None
        
        # Execute the action using tools
        try:
            observation, reward, done = self._execute_action(action)
        except Exception as e:
            logger.error(f"Error executing action '{action}': {e}")
            return None
        
        # Create new node
        new_state = {
            'thought': thought,
            'action': action,
            'observation': observation
        }
        
        new_node = Node(
            state=new_state,
            action=action,
            observation=observation,
            parent=parent,
            question=question
        )
        
        new_node.is_terminal = done or reward == 1
        new_node.reward = reward
        
        # Track failed trajectories for reflection
        if new_node.is_terminal and reward == 0:
            trajectory_text = new_node.get_trajectory_string()
            self.failed_trajectories.append({
                'trajectory': trajectory_text,
                'final_answer': action
            })
            
            # Generate reflections if enabled
            if (self.reflection_enabled and 
                len(self.failed_trajectories) >= self.reflection_config.get('min_failures_for_reflection', 1) and
                len(self.reflection_map) < self.max_reflections):
                
                try:
                    new_reflections = self.task.generate_self_reflection(
                        [t['trajectory'] for t in self.failed_trajectories[-3:]], 
                        question
                    )
                    self.reflection_map.extend(new_reflections)
                except Exception as e:
                    logger.error(f"Error generating reflections: {e}")
        
        logger.debug(f"Created new node: depth={new_node.depth}, reward={reward}, terminal={done}")
        
        return new_node
    
    def _execute_action(self, action: str) -> Tuple[str, float, bool]:
        """
        Execute an action using the registered tools.
        
        Args:
            action: Action string to execute
            
        Returns:
            Tuple of (observation, reward, done)
        """
        # Parse action format: tool_name[parameter]
        if '[' in action and ']' in action:
            tool_name = action.split('[')[0].lower()
            parameter = action.split('[')[1].split(']')[0]
        else:
            tool_name = action.lower()
            parameter = ""
        
        # Check if tool exists
        if tool_name not in self.tool_registry:
            return f"Error: Unknown tool '{tool_name}'", 0.0, False
        
        try:
            # Get and execute tool
            tool = self.tool_registry.get_tool(tool_name)
            
            if hasattr(tool, '_run'):
                # LangChain tool
                result = tool._run(parameter)
            elif callable(tool):
                # Function tool
                result = tool(parameter)
            else:
                return f"Error: Tool '{tool_name}' not callable", 0.0, False
            
            # Handle different tool results
            if tool_name == 'finish':
                # Terminal action - evaluate answer
                return result, 1.0, True  # Assume success, actual evaluation happens later
            elif tool_name in ['search', 'lookup']:
                # Continue with search/lookup result
                return result, 0.0, False
            else:
                # Generic tool result
                return result, 0.0, False
                
        except Exception as e:
            logger.error(f"Tool execution error for '{action}': {e}")
            return f"Error executing {tool_name}: {str(e)}", 0.0, False
    
    def _evaluate_node(self, node: Node, question: str) -> None:
        """
        Evaluate the children of a node using value estimation.
        
        Args:
            node: Node whose children to evaluate
            question: Original question
        """
        if not node.children:
            return
        
        non_terminal_children = [child for child in node.children if not child.is_terminal]
        
        if not non_terminal_children:
            return
        
        # Generate prompts for each child
        child_prompts = []
        for child in non_terminal_children:
            trajectory = child.get_trajectory_string()
            child_prompts.append(trajectory)
        
        # Get value estimates
        try:
            values = self._get_values(question, child_prompts)
            
            # Assign values to children
            for child, value in zip(non_terminal_children, values):
                child.value = value
                child.visits = 1
                
        except Exception as e:
            logger.error(f"Error evaluating node children: {e}")
            # Assign default values
            for child in non_terminal_children:
                child.value = 0.5
                child.visits = 1
    
    def _get_values(self, question: str, trajectories: List[str]) -> List[float]:
        """
        Get value estimates for trajectories.
        
        Args:
            question: Original question
            trajectories: List of trajectory strings
            
        Returns:
            List of value estimates
        """
        values = []
        
        for trajectory in trajectories:
            # Check cache first
            cache_key = f"{question}::{trajectory}"
            
            if self.value_cache and cache_key in self.value_cache:
                values.append(self.value_cache[cache_key])
                continue
            
            # Generate value estimation prompt
            value_prompt = self.task.value_prompt_wrap(
                question, 
                trajectory, 
                [t['trajectory'] for t in self.failed_trajectories[-3:]], 
                self.reflection_map
            )
            
            try:
                # Get value estimates from LLM
                value_outputs = self.llm.generate(
                    value_prompt,
                    n=self.n_evaluate_sample
                )
                
                # Parse value from outputs
                value = self.task.value_outputs_unwrap(value_outputs)
                
                # Cache the value
                if self.value_cache:
                    self.value_cache[cache_key] = value
                    
                values.append(value)
                
            except Exception as e:
                logger.error(f"Error getting value estimate: {e}")
                values.append(0.5)  # Default neutral value
        
        return values
    
    def _rollout(self, node: Node, question: str, max_depth: int = 4) -> Tuple[float, Node]:
        """
        Perform rollout simulation from a node.
        
        Args:
            node: Starting node for rollout
            question: Original question
            max_depth: Maximum rollout depth
            
        Returns:
            Tuple of (average reward, final node)
        """
        logger.debug(f"Starting rollout from depth {node.depth}")
        
        current = node
        depth = node.depth
        rewards = [0.0]
        
        while not current.is_terminal and depth < max_depth:
            # Generate new states for rollout
            new_states = self._generate_new_states(current, question)
            
            if not new_states:
                break
            
            # Check for terminal states
            terminal_states = [s for s in new_states if s.is_terminal]
            if terminal_states:
                best_terminal = max(terminal_states, key=lambda x: x.reward)
                return best_terminal.reward, best_terminal
            
            # Evaluate non-terminal states
            non_terminal_states = [s for s in new_states if not s.is_terminal]
            if non_terminal_states:
                # Get trajectory prompts
                trajectories = [s.get_trajectory_string() for s in non_terminal_states]
                values = self._get_values(question, trajectories)
                
                # Select best state
                max_value_idx = values.index(max(values))
                current = non_terminal_states[max_value_idx]
                rewards.append(max(values))
            else:
                break
            
            depth += 1
        
        if depth >= max_depth:
            rewards = [-0.1]  # Penalty for not reaching terminal state
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        
        logger.debug(f"Rollout completed: avg_reward={avg_reward:.3f}")
        
        return avg_reward, current
    
    def _backpropagate(self, node: Node, reward: float) -> None:
        """
        Backpropagate reward up the tree.
        
        Args:
            node: Starting node for backpropagation
            reward: Reward to propagate
        """
        current = node
        
        while current is not None:
            current.update_value(reward, visits=1)
            current = current.parent
    
    def _collect_all_nodes(self, root: Node) -> List[Node]:
        """
        Collect all nodes in the tree.
        
        Args:
            root: Root node
            
        Returns:
            List of all nodes
        """
        nodes = [root]
        
        def collect_recursive(node):
            for child in node.children:
                nodes.append(child)
                collect_recursive(child)
        
        collect_recursive(root)
        return nodes
    
    def _extract_answer(self, node: Node) -> str:
        """
        Extract the final answer from a node's trajectory.
        
        Args:
            node: Node to extract answer from
            
        Returns:
            Extracted answer string
        """
        trajectory = node.get_trajectory()
        
        # Look for finish actions
        for step in reversed(trajectory):
            action = step.get('action', '')
            if action.lower().startswith('finish['):
                # Extract answer from finish[answer]
                if '[' in action and ']' in action:
                    return action.split('[')[1].split(']')[0]
        
        # Fallback: use last observation or action
        if trajectory:
            last_step = trajectory[-1]
            return last_step.get('observation', last_step.get('action', 'No answer'))
        
        return 'No answer found'
    
    def _tree_to_dict(self, root: Node) -> Dict[str, Any]:
        """
        Convert search tree to dictionary representation.
        
        Args:
            root: Root node of the tree
            
        Returns:
            Dictionary representation of the tree
        """
        def node_to_dict(node):
            return {
                'id': node.id,
                'depth': node.depth,
                'action': node.action,
                'observation': node.observation[:100] + '...' if len(node.observation) > 100 else node.observation,
                'value': node.value,
                'visits': node.visits,
                'reward': node.reward,
                'is_terminal': node.is_terminal,
                'children': [node_to_dict(child) for child in node.children]
            }
        
        return node_to_dict(root)
    
    def visualize_tree(self, tree_dict: Dict[str, Any], output_path: str) -> None:
        """
        Create a visualization of the search tree.
        
        Args:
            tree_dict: Dictionary representation of the tree
            output_path: Path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from pathlib import Path
            
            # Create directed graph
            G = nx.DiGraph()
            pos = {}
            labels = {}
            
            def add_nodes(node_dict, parent_id=None, x=0, y=0, level=0):
                node_id = node_dict['id']
                G.add_node(node_id)
                
                # Position nodes
                pos[node_id] = (x, -level)
                
                # Create label
                action = node_dict['action'][:20] + '...' if len(node_dict['action']) > 20 else node_dict['action']
                labels[node_id] = f"{action}\nv:{node_dict['value']:.2f}"
                
                if parent_id:
                    G.add_edge(parent_id, node_id)
                
                # Add children
                children = node_dict['children']
                if children:
                    child_spacing = max(1, 10 / (level + 1))
                    start_x = x - (len(children) - 1) * child_spacing / 2
                    
                    for i, child in enumerate(children):
                        child_x = start_x + i * child_spacing
                        add_nodes(child, node_id, child_x, y, level + 1)
            
            # Build graph
            add_nodes(tree_dict)
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            nx.draw(G, pos, labels=labels, with_labels=True, 
                   node_color='lightblue', node_size=1000, 
                   font_size=8, arrows=True)
            
            plt.title("LATS Search Tree")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Tree visualization saved to {output_path}")
            
        except ImportError:
            logger.warning("matplotlib or networkx not available for tree visualization")
        except Exception as e:
            logger.error(f"Error creating tree visualization: {e}")