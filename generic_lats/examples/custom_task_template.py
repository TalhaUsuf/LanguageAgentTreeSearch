"""
Template for Creating Custom Tasks with Generic LATS

This template shows how to:
1. Define custom tools using LangChain decorators
2. Create a custom task class
3. Configure prompts for your domain
4. Run LATS with your custom setup

Usage:
    python custom_task_template.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from langchain.tools import tool
from tasks.base_task import BaseTask
from tools.tool_registry import ToolRegistry
from tools.base_tools import FinishTool
from core.lats_algorithm import LATSSearch
from models.llm_wrapper import LangChainLLM
from utils.config_loader import ConfigLoader


# ============================================================================
# Step 1: Define Custom Tools
# ============================================================================

@tool
def database_query(query: str) -> str:
    """
    Execute a database query and return results.
    
    Args:
        query: SQL query or database query string
        
    Returns:
        Query results as formatted string
    """
    # Simulated database - replace with your actual implementation
    mock_database = {
        "employees": [
            {"id": 1, "name": "Alice", "department": "Engineering", "salary": 85000},
            {"id": 2, "name": "Bob", "department": "Marketing", "salary": 75000},
            {"id": 3, "name": "Carol", "department": "Engineering", "salary": 90000},
            {"id": 4, "name": "Dave", "department": "Sales", "salary": 70000},
        ],
        "departments": [
            {"name": "Engineering", "budget": 500000, "location": "Building A"},
            {"name": "Marketing", "budget": 300000, "location": "Building B"},
            {"name": "Sales", "budget": 250000, "location": "Building C"},
        ]
    }
    
    try:
        # Simple query parsing (replace with actual SQL engine)
        query_lower = query.lower()
        
        if "employees" in query_lower:
            if "engineering" in query_lower:
                results = [emp for emp in mock_database["employees"] if emp["department"] == "Engineering"]
            elif "salary > 80000" in query_lower:
                results = [emp for emp in mock_database["employees"] if emp["salary"] > 80000]
            else:
                results = mock_database["employees"]
            
            return f"Found {len(results)} employees: " + ", ".join([emp["name"] for emp in results])
        
        elif "departments" in query_lower:
            results = mock_database["departments"]
            return f"Found {len(results)} departments: " + ", ".join([dept["name"] for dept in results])
        
        else:
            return f"Query executed: {query} (No specific results - implement your logic here)"
            
    except Exception as e:
        return f"Query error: {str(e)}"


@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations safely.
    
    Args:
        expression: Mathematical expression to evaluate
        
    Returns:
        Result of the calculation
    """
    try:
        # Basic safety check
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        
        result = eval(expression)
        return f"Result: {result}"
        
    except Exception as e:
        return f"Calculation error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """
    Simulated web search tool.
    
    Args:
        query: Search query
        
    Returns:
        Search results
    """
    # Mock search results - replace with actual web search API
    mock_results = {
        "python": "Python is a high-level programming language known for its simplicity and readability.",
        "machine learning": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
        "database": "A database is a structured collection of data stored electronically in a computer system.",
        "api": "An API (Application Programming Interface) is a set of protocols for building software applications."
    }
    
    query_lower = query.lower()
    
    # Find matching results
    results = []
    for key, value in mock_results.items():
        if key in query_lower:
            results.append(f"{key.title()}: {value}")
    
    if results:
        return "Search results:\n" + "\n".join(results)
    else:
        return f"No results found for '{query}'. Try searching for: python, machine learning, database, api"


@tool
def file_analyzer(file_path: str) -> str:
    """
    Analyze a file and return information about it.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        File analysis results
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return f"File not found: {file_path}"
        
        # Basic file analysis
        stats = path.stat()
        analysis = f"File: {path.name}\n"
        analysis += f"Size: {stats.st_size} bytes\n"
        analysis += f"Type: {path.suffix if path.suffix else 'No extension'}\n"
        
        # Try to read content if it's a text file
        if path.suffix in ['.txt', '.py', '.md', '.json', '.yaml', '.yml']:
            try:
                content = path.read_text(encoding='utf-8')
                lines = len(content.splitlines())
                chars = len(content)
                analysis += f"Lines: {lines}\n"
                analysis += f"Characters: {chars}\n"
                
                # Show first few lines
                first_lines = content.splitlines()[:3]
                analysis += "First few lines:\n" + "\n".join(first_lines)
            except Exception as e:
                analysis += f"Could not read content: {e}"
        
        return analysis
        
    except Exception as e:
        return f"File analysis error: {str(e)}"


# ============================================================================
# Step 2: Define Custom Task Class  
# ============================================================================

class CustomReasoningTask(BaseTask):
    """
    Custom reasoning task that combines database queries, calculations, and web search.
    
    This demonstrates how to create a task that requires multi-step reasoning
    across different types of tools and data sources.
    """
    
    def __init__(self, data_source: Optional[str] = None, **kwargs):
        """
        Initialize custom task.
        
        Args:
            data_source: Path to data source (optional)
            **kwargs: Additional arguments
        """
        super().__init__(**kwargs)
        
        self.data_source = data_source
        self.data = self._load_custom_data()
        
    def _load_custom_data(self) -> List[Dict[str, Any]]:
        """Load custom task data."""
        # Sample reasoning tasks - replace with your actual data
        return [
            {
                "id": "reasoning_1",
                "question": "How many employees in the Engineering department earn more than the average salary of all employees?",
                "answer": "2",
                "type": "database_reasoning",
                "difficulty": "medium"
            },
            {
                "id": "reasoning_2", 
                "question": "If the Marketing department's budget is increased by 15%, what would be the new total budget across all departments?",
                "answer": "1095000",
                "type": "calculation_reasoning",
                "difficulty": "medium"
            },
            {
                "id": "reasoning_3",
                "question": "What is machine learning and how does it relate to Python programming?",
                "answer": "Machine learning is a subset of AI that enables computers to learn without explicit programming. Python is commonly used for machine learning due to its simplicity, extensive libraries, and strong community support.",
                "type": "knowledge_synthesis",
                "difficulty": "easy"
            },
            {
                "id": "reasoning_4",
                "question": "Calculate the average salary of Engineering employees and compare it to the Marketing department budget per employee if Marketing has 5 employees.",
                "answer": "87500 vs 60000",
                "type": "multi_step_calculation",
                "difficulty": "hard"
            }
        ]
    
    def initialize_task(self):
        """Initialize custom task prompts and settings."""
        if not self.prompts:
            self.prompts = {
                'cot_prompt': self._get_cot_prompt(),
                'value_prompt': self._get_value_prompt(),
                'reflection_prompt': self._get_reflection_prompt()
            }
    
    def get_input(self, idx: int) -> str:
        """Get input question for task index."""
        if idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for {len(self.data)} questions")
        return self.data[idx]['question']
    
    def evaluate_output(self, idx: int, output: str) -> Dict[str, Any]:
        """Evaluate output against ground truth."""
        if idx >= len(self.data):
            return {'is_correct': False, 'score': 0.0, 'error': 'Index out of range'}
        
        ground_truth = self.data[idx]['answer']
        
        # Flexible matching for different answer formats
        output_clean = output.strip().lower()
        ground_truth_clean = ground_truth.strip().lower()
        
        # Check for exact match
        exact_match = output_clean == ground_truth_clean
        
        # Check for partial match (contains key information)
        partial_match = ground_truth_clean in output_clean or output_clean in ground_truth_clean
        
        # For numerical answers, try to extract and compare numbers
        score = 1.0 if exact_match else (0.7 if partial_match else 0.0)
        
        return {
            'is_correct': exact_match,
            'score': score,
            'exact_match': exact_match,
            'partial_match': partial_match,
            'ground_truth': ground_truth,
            'prediction': output,
            'question_id': self.data[idx]['id'],
            'question_type': self.data[idx]['type'],
            'difficulty': self.data[idx]['difficulty']
        }
    
    def get_task_description(self) -> str:
        """Get task description for the LLM."""
        return """
        You are solving custom reasoning tasks that may require:
        1. Database queries to retrieve information about employees and departments
        2. Mathematical calculations to analyze data
        3. Web searches to gather external knowledge
        4. File analysis to understand data sources
        
        Use the available tools systematically to gather information, perform calculations,
        and synthesize knowledge to answer complex questions accurately.
        
        Think step by step and use multiple tools when necessary to build a complete answer.
        """
    
    def _get_cot_prompt(self) -> str:
        """Get chain-of-thought prompt for custom reasoning."""
        return """
Solve the reasoning task step by step using the available tools.

Available Actions:
- database_query[query]: Execute database queries to get information about employees and departments
- calculate[expression]: Perform mathematical calculations
- web_search[query]: Search for information online
- file_analyzer[path]: Analyze files for additional context
- finish[answer]: Submit your final answer

Think through each step and use the appropriate tools to gather and process information.

Example:
Question: How many Engineering employees earn more than 80000?
Thought 1: I need to query the database to find Engineering employees with salary > 80000.
Action 1: database_query[employees in Engineering with salary > 80000]
Observation 1: Found 2 employees: Alice, Carol
Thought 2: The query returned 2 employees, so the answer is 2.
Action 2: finish[2]

Question: {input}
Current progress:
{trajectory}
"""
    
    def _get_value_prompt(self) -> str:
        """Get value evaluation prompt."""
        return """
Evaluate the current progress on this reasoning task.

Question: {question}
Current state: {state}
Progress so far: {trajectory}

Consider:
1. How much relevant information has been gathered
2. Whether the right tools are being used for the task type
3. How close the current state is to a complete answer
4. If the reasoning approach is logical and systematic

Rate the likelihood of success (0-10):
"""
    
    def _get_reflection_prompt(self) -> str:
        """Get reflection prompt for learning from failures."""
        return """
Analyze this failed reasoning attempt and suggest improvements.

Question: {question}
Failed attempt: {trajectory}
Correct answer: {correct_answer}

In 2-3 sentences, diagnose why this approach failed and suggest a better strategy:
"""
    
    def __len__(self) -> int:
        """Get number of questions in the dataset."""
        return len(self.data)
    
    def get_question_info(self, idx: int) -> Dict[str, Any]:
        """Get detailed information about a question."""
        if idx >= len(self.data):
            return {}
        return self.data[idx].copy()


# ============================================================================
# Step 3: Main Execution Function
# ============================================================================

def run_custom_task():
    """
    Main function demonstrating custom task with LATS.
    """
    print("🚀 Custom Task Template with Generic LATS")
    print("=" * 50)
    
    try:
        # Configuration for custom task
        config = {
            'llm': {
                'provider': 'openai-compatible',
                'model': 'llama-3.1-70b',
                'base_url': 'http://your-llm-server:port/v1',
                'api_key': '',
                'temperature': 1.0,
                'max_tokens': 500
            },
            'algorithm': {
                'type': 'lats',
                'iterations': 20,
                'n_generate_sample': 3,
                'n_evaluate_sample': 1,
                'max_depth': 8,
                'cache_values': True,
                'reflection': {
                    'enabled': True,
                    'max_reflections': 2
                }
            }
        }
        
        # Initialize LLM
        print("🤖 Initializing LLM...")
        llm = LangChainLLM(config['llm'])
        
        # Setup custom tools
        print("🛠️  Setting up custom tools...")
        tool_registry = ToolRegistry()
        
        # Register custom tools
        tool_registry.register("database_query", database_query)
        tool_registry.register("calculate", calculate)
        tool_registry.register("web_search", web_search)
        tool_registry.register("file_analyzer", file_analyzer)
        tool_registry.register("finish", FinishTool())
        
        print(f"✅ Registered {len(tool_registry)} tools")
        
        # Initialize custom task
        print("📋 Initializing custom reasoning task...")
        task = CustomReasoningTask(
            tool_registry=tool_registry,
            llm=llm,
            config=config
        )
        
        print(f"✅ Loaded {len(task)} custom reasoning questions")
        
        # Initialize LATS
        print("🔍 Initializing LATS search...")
        lats = LATSSearch(task, llm, tool_registry, config['algorithm'])
        
        # Process multiple examples
        results = []
        print(f"\n🎯 Solving {len(task)} reasoning tasks...")
        
        for idx in range(len(task)):
            print(f"\n{'='*20} Question {idx+1} {'='*20}")
            
            question = task.get_input(idx)
            question_info = task.get_question_info(idx)
            
            print(f"Question: {question}")
            print(f"Type: {question_info['type']}")
            print(f"Difficulty: {question_info['difficulty']}")
            
            # Run LATS search
            result = lats.search(question, idx)
            results.append(result)
            
            # Display result
            success = "✅" if result['is_correct'] else "❌"
            print(f"\n{success} Result: {result['answer']}")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Score: {result['score']:.2f}")
            print(f"Time: {result['search_time']:.1f}s")
            
            # Show key reasoning steps
            if result['trajectory']:
                print("\n🔍 Key Steps:")
                for i, step in enumerate(result['trajectory'][-3:], 1):  # Last 3 steps
                    if step.get('action'):
                        print(f"  {i}. {step['action']}")
        
        # Overall results
        print(f"\n{'='*50}")
        print("📊 OVERALL RESULTS")
        print(f"{'='*50}")
        
        total_correct = sum(1 for r in results if r['is_correct'])
        avg_score = sum(r['score'] for r in results) / len(results)
        avg_time = sum(r['search_time'] for r in results) / len(results)
        
        print(f"Questions processed: {len(results)}")
        print(f"Correct answers: {total_correct}")
        print(f"Accuracy: {total_correct/len(results):.1%}")
        print(f"Average score: {avg_score:.3f}")
        print(f"Average time: {avg_time:.1f} seconds")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_custom_config_template():
    """Create a template configuration file for custom tasks."""
    
    config_template = """
# Custom Task Configuration Template
# Modify this template for your specific use case

llm:
  provider: "openai-compatible"
  base_url: "${OPENAI_API_BASE}"
  model: "${OPENAI_MODEL}"
  temperature: 1.0
  max_tokens: 500

algorithm:
  type: "lats"
  iterations: 25
  n_generate_sample: 4
  n_evaluate_sample: 1
  max_depth: 10
  cache_values: true
  reflection:
    enabled: true
    max_reflections: 3

task:
  type: "custom"
  data_source: "path/to/your/data.json"
  params:
    custom_param1: "value1"
    custom_param2: "value2"

tools:
  enabled_tools:
    - "your_custom_tool1"
    - "your_custom_tool2" 
    - "finish"
  
  custom_tools_dir: "tools/custom_tools"

logging:
  level: "INFO"
  file: "logs/custom_task.log"

output:
  save_trajectories: true
  trajectories_dir: "output/trajectories"
"""
    
    config_file = Path("custom_task_config.yaml")
    config_file.write_text(config_template.strip())
    print(f"📄 Created config template: {config_file}")


if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--create-config":
            create_custom_config_template()
        elif sys.argv[1] == "--help":
            print("Custom Task Template Usage:")
            print("  python custom_task_template.py           # Run the demo")
            print("  python custom_task_template.py --create-config  # Create config template") 
            print("  python custom_task_template.py --help    # Show this help")
    else:
        run_custom_task()