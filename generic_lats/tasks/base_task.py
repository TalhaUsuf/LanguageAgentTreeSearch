"""
Abstract base task class for Generic LATS framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """
    Abstract base class for all tasks in the Generic LATS framework.
    
    This class defines the interface that all task implementations must follow.
    It provides common functionality for loading prompts, managing tools, and
    evaluating outputs.
    """
    
    def __init__(self, 
                 tool_registry: Any = None,
                 llm: Any = None,
                 config: Dict[str, Any] = None,
                 prompts_config: Optional[str] = None):
        """
        Initialize the base task.
        
        Args:
            tool_registry: Registry containing available tools
            llm: LLM instance for generation
            config: Task configuration dictionary
            prompts_config: Path to prompts configuration file
        """
        self.tool_registry = tool_registry
        self.llm = llm
        self.config = config or {}
        self.prompts = {}
        
        # Load prompts if configuration provided
        if prompts_config:
            self.load_prompts(prompts_config)
        elif config and 'prompts_config' in config:
            self.load_prompts(config['prompts_config'])
        
        # Task-specific initialization
        self.initialize_task()
        
        logger.info(f"Initialized task: {self.__class__.__name__}")
    
    def initialize_task(self):
        """Initialize task-specific components. Override in subclasses."""
        pass
    
    def load_prompts(self, prompts_path: str):
        """
        Load prompt templates from YAML file.
        
        Args:
            prompts_path: Path to prompts YAML file
        """
        prompts_path = Path(prompts_path)
        if prompts_path.exists():
            with open(prompts_path, 'r', encoding='utf-8') as f:
                prompts_data = yaml.safe_load(f)
                self.prompts = prompts_data.get('prompts', {})
            logger.info(f"Loaded prompts from: {prompts_path}")
        else:
            logger.warning(f"Prompts file not found: {prompts_path}")
    
    @abstractmethod
    def get_input(self, idx: int) -> str:
        """
        Get the input for a specific task instance.
        
        Args:
            idx: Index of the task instance
            
        Returns:
            Input string for the task
        """
        pass
    
    @abstractmethod
    def evaluate_output(self, idx: int, output: str) -> Dict[str, Any]:
        """
        Evaluate the output against ground truth.
        
        Args:
            idx: Index of the task instance
            output: Generated output to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_task_description(self) -> str:
        """
        Get a description of the task for the LLM.
        
        Returns:
            Task description string
        """
        pass
    
    def get_tools_description(self) -> str:
        """
        Get a description of available tools.
        
        Returns:
            Tools description string
        """
        if not self.tool_registry:
            return "No tools available."
        
        tool_descriptions = []
        for tool_name in self.tool_registry.get_tool_names():
            tool = self.tool_registry.get_tool(tool_name)
            if hasattr(tool, 'description'):
                tool_descriptions.append(f"- {tool_name}: {tool.description}")
            else:
                tool_descriptions.append(f"- {tool_name}: No description available")
        
        return "\n".join(tool_descriptions)
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """
        Format a prompt template with given variables.
        
        Args:
            template_name: Name of the prompt template
            **kwargs: Variables to substitute in the template
            
        Returns:
            Formatted prompt string
        """
        if template_name not in self.prompts:
            raise ValueError(f"Prompt template '{template_name}' not found")
        
        template = self.prompts[template_name]
        
        # Add common variables
        kwargs.setdefault('tools_description', self.get_tools_description())
        kwargs.setdefault('task_description', self.get_task_description())
        
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Missing variable {e} for prompt template '{template_name}'")
            raise
    
    def standard_prompt_wrap(self, x: str, y: str = '') -> str:
        """
        Wrap input with standard prompt template.
        
        Args:
            x: Input text
            y: Optional continuation text
            
        Returns:
            Formatted prompt
        """
        if 'standard_prompt' in self.prompts:
            return self.format_prompt('standard_prompt', input=x) + y
        else:
            # Fallback to simple concatenation
            return f"Task: {x}\n{y}"
    
    def cot_prompt_wrap(self, x: str, y: str = '', reflection_mapping_list: List[Dict] = None) -> str:
        """
        Wrap input with chain-of-thought prompt template with dynamic example selection.
        
        Args:
            x: Input text
            y: Optional trajectory text
            reflection_mapping_list: List of reflection mappings
            
        Returns:
            Formatted prompt
        """
        # Prepare trajectory text with reflections
        trajectory_text = y
        if reflection_mapping_list:
            reflection_text = ""
            for reflection in reflection_mapping_list:
                reflection_text += f"Previous failed attempt:\n{reflection.get('trajectory', '')}\n"
                reflection_text += f"Reflection: {reflection.get('reflection', '')}\n\n"
            trajectory_text = reflection_text + y
        
        # Generate examples dynamically based on available tools
        examples_text = self._generate_relevant_examples()
        
        if 'cot_prompt' in self.prompts:
            return self.format_prompt(
                'cot_prompt',
                input=x,
                trajectory=trajectory_text,
                examples=examples_text,
                tools=self.get_tools_description()
            )
        else:
            # Fallback prompt
            return f"""Solve the following task step by step:

Available tools:
{self.get_tools_description()}

Task: {x}
Current progress:
{trajectory_text}

Provide your next thought and action:"""
    
    def _generate_relevant_examples(self) -> str:
        """
        Generate relevant examples based on available tools.
        
        Returns:
            Formatted examples string
        """
        if not self.tool_registry:
            return self._get_fallback_examples()
        
        available_tools = self.tool_registry.get_tool_names()
        selected_examples = []
        
        # Get examples from prompts configuration
        if 'examples' in self.prompts:
            examples = self.prompts['examples']
            if isinstance(examples, list):
                # Select examples that match available tool patterns
                for example in examples:
                    if isinstance(example, dict):
                        pattern = example.get('pattern', '')
                        trajectory = example.get('trajectory', '')
                        
                        # Check if this example's tools are available
                        if self._example_matches_tools(trajectory, available_tools):
                            question = example.get('question', '')
                            selected_examples.append(f"Question: {question}\n{trajectory}\n")
        
        # If we have fewer than 2 examples, add some generic ones
        if len(selected_examples) < 2:
            selected_examples.extend(self._get_generic_examples_for_tools(available_tools))
        
        # Limit to 3 examples to avoid overly long prompts
        return "\n".join(selected_examples[:3])
    
    def _example_matches_tools(self, trajectory: str, available_tools: List[str]) -> bool:
        """
        Check if an example trajectory uses tools that are available.
        
        Args:
            trajectory: Example trajectory text
            available_tools: List of available tool names
            
        Returns:
            True if example uses available tools
        """
        # Extract tool names from trajectory (look for Action lines)
        import re
        action_pattern = r'Action \d+: (\w+)\['
        used_tools = re.findall(action_pattern, trajectory)
        
        # Check if at least 70% of used tools are available
        if not used_tools:
            return True  # No tools used, example is generic
        
        available_count = sum(1 for tool in used_tools if tool.lower() in [t.lower() for t in available_tools])
        return available_count / len(used_tools) >= 0.7
    
    def _get_generic_examples_for_tools(self, available_tools: List[str]) -> List[str]:
        """
        Generate generic examples based on available tool types.
        
        Args:
            available_tools: List of available tool names
            
        Returns:
            List of example strings
        """
        examples = []
        
        # Database example if DB tools available
        if any('database' in tool.lower() or 'db' in tool.lower() for tool in available_tools):
            examples.append("""Question: How many items are in the inventory?
Thought 1: I need to query the database to count inventory items.
Action 1: database_query[SELECT COUNT(*) FROM inventory WHERE status='active']
Observation 1: Query returned: 1,247 active items in inventory
Thought 2: The inventory contains 1,247 active items.
Action 2: finish[1,247 items]
""")
        
        # API example if API tools available
        if any('api' in tool.lower() for tool in available_tools):
            examples.append("""Question: What is the current status of order #12345?
Thought 1: I need to check the order status using the API.
Action 1: order_api[get_status, order_id=12345]
Observation 1: Order #12345 status: Shipped, Tracking: ABC123, Expected delivery: 2024-10-28
Thought 2: The order has been shipped with tracking number ABC123.
Action 2: finish[Order #12345 is shipped with tracking ABC123, expected delivery Oct 28, 2024]
""")
        
        # Calculation example if math tools available
        if any('calc' in tool.lower() or 'math' in tool.lower() for tool in available_tools):
            examples.append("""Question: What is the total cost of 25 items at $12.99 each plus 8.5% tax?
Thought 1: I need to calculate the subtotal first, then add tax.
Action 1: calculate[25 * 12.99]
Observation 1: Subtotal: $324.75
Thought 2: Now I need to calculate the tax amount and add it.
Action 2: calculate[324.75 * 0.085]
Observation 2: Tax amount: $27.60
Thought 3: Now I can calculate the total.
Action 3: calculate[324.75 + 27.60]
Observation 3: Total: $352.35
Action 4: finish[$352.35]
""")
        
        return examples
    
    def _get_fallback_examples(self) -> str:
        """Get basic fallback examples when no tool registry is available."""
        return """Question: What is the square root of 144?
Thought 1: I need to calculate the square root of 144.
Action 1: calculate[sqrt(144)]
Observation 1: The square root of 144 is 12
Action 2: finish[12]

Question: Find information about Python programming.
Thought 1: I should search for information about Python programming.
Action 1: search[Python programming language]
Observation 1: Python is a high-level programming language known for its simplicity and readability...
Thought 2: I have found relevant information about Python.
Action 2: finish[Python is a high-level programming language known for its simplicity and readability]
"""
    
    def value_prompt_wrap(self, x: str, y: str, failed_trajectories: List[str] = None, 
                         reflection_map: List[Dict] = None) -> str:
        """
        Wrap input with value evaluation prompt template.
        
        Args:
            x: Input text (question)
            y: Current state/trajectory
            failed_trajectories: List of failed trajectory texts
            reflection_map: List of reflection mappings
            
        Returns:
            Formatted value evaluation prompt
        """
        # Add context from failed trajectories and reflections
        context = ""
        if failed_trajectories:
            context += "Previous failed attempts:\n"
            for i, traj in enumerate(failed_trajectories[:3]):  # Limit to 3
                context += f"Attempt {i+1}: {traj}\n"
        
        if reflection_map:
            context += "\nReflections on failures:\n"
            for reflection in reflection_map:
                context += f"- {reflection.get('reflection', '')}\n"
        
        full_state = f"{context}\n\nCurrent state: {y}" if context else y
        
        if 'value_prompt' in self.prompts:
            return self.format_prompt(
                'value_prompt',
                question=x,
                state=full_state,
                trajectory=y
            )
        else:
            # Fallback value prompt
            return f"""Evaluate how promising the current state is for solving this question.

Question: {x}
{full_state}

Rate the likelihood of success from this state (0-10 where 10 is most likely to succeed):"""
    
    def value_outputs_unwrap(self, value_outputs: List[str]) -> float:
        """
        Extract numerical value from LLM value evaluation outputs.
        
        Args:
            value_outputs: List of value evaluation strings from LLM
            
        Returns:
            Average numerical value (0.0-1.0)
        """
        values = []
        
        for output in value_outputs:
            try:
                # Try to extract number from output
                import re
                numbers = re.findall(r'\b\d+(?:\.\d+)?\b', output)
                
                if numbers:
                    # Take the first number found
                    value = float(numbers[0])
                    # Normalize to 0-1 range if it seems to be on 0-10 scale
                    if value > 1.0:
                        value = value / 10.0
                    values.append(min(1.0, max(0.0, value)))
                else:
                    # Look for qualitative indicators
                    output_lower = output.lower()
                    if any(word in output_lower for word in ['high', 'good', 'promising']):
                        values.append(0.8)
                    elif any(word in output_lower for word in ['low', 'poor', 'unlikely']):
                        values.append(0.2)
                    else:
                        values.append(0.5)  # Neutral default
                        
            except Exception as e:
                logger.warning(f"Error parsing value output '{output}': {e}")
                values.append(0.5)  # Default neutral value
        
        return sum(values) / len(values) if values else 0.5
    
    def generate_self_reflection(self, failed_trajectories: List[str], question: str) -> List[Dict[str, str]]:
        """
        Generate self-reflection on failed trajectories.
        
        Args:
            failed_trajectories: List of failed trajectory strings
            question: The original question/task
            
        Returns:
            List of reflection dictionaries
        """
        reflections = []
        
        if not self.llm or 'reflection_prompt' not in self.prompts:
            return reflections
        
        # Limit to avoid too much context
        trajectories_to_reflect = failed_trajectories[:3]
        
        for trajectory in trajectories_to_reflect:
            try:
                reflection_prompt = self.format_prompt(
                    'reflection_prompt',
                    trajectory=trajectory,
                    question=question,
                    answer="Failed to find correct answer",
                    correct_answer="Unknown"
                )
                
                reflection_outputs = self.llm.generate(reflection_prompt, n=1)
                
                if reflection_outputs:
                    reflections.append({
                        'question': question,
                        'trajectory': trajectory,
                        'reflection': reflection_outputs[0]
                    })
                    
            except Exception as e:
                logger.error(f"Error generating reflection: {e}")
        
        return reflections
    
    def __len__(self) -> int:
        """
        Get the number of task instances.
        
        Returns:
            Number of instances in the task
        """
        # Default implementation returns 0, override in subclasses
        return 0
    
    def get_metrics_names(self) -> List[str]:
        """
        Get list of metrics that this task can evaluate.
        
        Returns:
            List of metric names
        """
        return ['accuracy', 'score']