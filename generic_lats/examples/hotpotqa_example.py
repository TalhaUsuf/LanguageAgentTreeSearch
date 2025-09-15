"""
Complete HotPotQA Example using Generic LATS Framework

This example demonstrates the complete workflow:
1. Loading configuration
2. Setting up tools and LLM
3. Initializing the HotPotQA task
4. Running the LATS algorithm
5. Processing and displaying results

Usage:
    python hotpotqa_example.py --config config/examples/hotpotqa_config.yaml
    python hotpotqa_example.py --task-index 5 --verbose
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.lats_algorithm import LATSSearch
from tasks.hotpotqa_task import HotPotQATask
from tools.tool_registry import ToolRegistry
from tools.wikipedia_tools import WikipediaSearchTool, WikipediaLookupTool
from tools.base_tools import FinishTool
from models.llm_wrapper import LangChainLLM
from utils.config_loader import ConfigLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(config):
    """Setup logging based on configuration."""
    log_config = config.get('logging', {})
    
    # Create logs directory
    log_file = log_config.get('file', 'logs/hotpotqa_example.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    level = getattr(logging, log_config.get('level', 'INFO').upper())
    formatter = logging.Formatter(log_config.get('format', 
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Console handler  
    console_config = log_config.get('console', {})
    if console_config.get('enabled', True):
        console_handler = logging.StreamHandler()
        console_level = getattr(logging, console_config.get('level', 'INFO').upper())
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.handlers = []  # Clear existing handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(level)


def main():
    """Main function demonstrating the complete LATS workflow."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="HotPotQA LATS Example")
    parser.add_argument(
        "--config", 
        default="config/examples/hotpotqa_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Index of the task to solve (0-based)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of LATS iterations (overrides config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files"
    )
    args = parser.parse_args()
    
    try:
        # Step 1: Load configuration
        print("🔧 Loading configuration...")
        config = ConfigLoader.load(args.config)
        
        # Override config with command line args
        if args.verbose:
            config['logging']['level'] = 'DEBUG'
            config['logging']['console']['level'] = 'DEBUG'
        
        # Setup logging
        setup_logging(config)
        logger.info("Starting HotPotQA LATS Example")
        logger.info(f"Configuration loaded from: {args.config}")
        
        # Step 2: Initialize LLM
        print("🤖 Initializing LLM...")
        logger.info("Initializing LLM with configuration")
        llm = LangChainLLM(config['llm'])
        logger.info(f"LLM initialized: {config['llm']['provider']} - {config['llm']['model']}")
        
        # Step 3: Setup tools
        print("🛠️  Setting up tools...")
        logger.info("Setting up tool registry")
        tool_registry = ToolRegistry()
        
        # Register Wikipedia tools with enhanced Wikipedia environment coordination
        search_tool = WikipediaSearchTool()
        lookup_tool = WikipediaLookupTool()
        finish_tool = FinishTool()
        
        tool_registry.register("search", search_tool, "Search Wikipedia for entities")
        tool_registry.register("lookup", lookup_tool, "Look up keywords in current page")
        tool_registry.register("finish", finish_tool, "Submit final answer")
        
        logger.info(f"Registered {len(tool_registry)} tools: {list(tool_registry.get_tool_names())}")
        
        # Step 4: Initialize HotPotQA task
        print("📚 Initializing HotPotQA task...")
        logger.info("Initializing HotPotQA task")
        
        # Get dataset path relative to current directory
        dataset_path = Path(config['task']['dataset']['path'])
        if not dataset_path.is_absolute():
            # Try relative to script directory
            script_dir = Path(__file__).parent
            dataset_path = script_dir / dataset_path
            if not dataset_path.exists():
                # Try relative to working directory
                dataset_path = Path.cwd() / config['task']['dataset']['path']
        
        if not dataset_path.exists():
            logger.error(f"Dataset not found at {dataset_path}")
            print(f"❌ Error: Dataset not found at {dataset_path}")
            print("Please check the dataset path in your configuration file.")
            return 1
        
        # Load prompts configuration if specified
        prompts_config = config.get('prompts', {}).get('config_file')
        if prompts_config and not Path(prompts_config).is_absolute():
            script_dir = Path(__file__).parent
            prompts_config = str(script_dir / prompts_config)
        
        task = HotPotQATask(
            dataset_path=str(dataset_path),
            split=config['task']['dataset']['split'],
            tool_registry=tool_registry,
            llm=llm,
            config=config,
            prompts_config=prompts_config
        )
        
        logger.info(f"Task initialized with {len(task)} questions")
        print(f"📊 Loaded {len(task)} questions from HotPotQA dataset")
        
        # Validate task index
        if args.task_index >= len(task):
            logger.error(f"Task index {args.task_index} out of range (0-{len(task)-1})")
            print(f"❌ Error: Task index {args.task_index} out of range (0-{len(task)-1})")
            return 1
        
        # Step 5: Get the question to solve
        question = task.get_input(args.task_index)
        question_info = task.get_question_info(args.task_index)
        
        print(f"\n📝 Question {args.task_index}:")
        print(f"   ID: {question_info.get('id', 'unknown')}")
        print(f"   Level: {question_info.get('level', 'unknown')}")
        print(f"   Type: {question_info.get('type', 'unknown')}")
        print(f"   Question: {question}")
        print(f"   Ground Truth: {question_info.get('answer', 'unknown')}")
        
        logger.info(f"Solving question {args.task_index}: {question[:100]}...")
        
        # Step 6: Initialize LATS search
        print("\n🔍 Initializing LATS search...")
        logger.info("Initializing LATS search algorithm")
        
        lats = LATSSearch(
            task=task,
            llm=llm,
            tool_registry=tool_registry,
            config=config['algorithm']
        )
        
        # Step 7: Run the search
        print("🚀 Starting LATS search...")
        logger.info("Starting LATS search process")
        
        # Determine iterations
        iterations = args.iterations if args.iterations else config['algorithm']['iterations']
        logger.info(f"Running search with {iterations} iterations")
        
        result = lats.search(
            question=question,
            task_index=args.task_index,
            iterations=iterations
        )
        
        # Step 8: Process and display results
        print("\n" + "="*60)
        print("🎯 SEARCH RESULTS")
        print("="*60)
        
        success_emoji = "✅" if result['is_correct'] else "❌"
        print(f"{success_emoji} Result: {'CORRECT' if result['is_correct'] else 'INCORRECT'}")
        print(f"📝 Question: {question}")
        print(f"🤖 Predicted Answer: {result['answer']}")
        print(f"✅ Ground Truth: {result['ground_truth']}")
        print(f"📊 Score (F1): {result['score']:.4f}")
        print(f"🎯 Exact Match: {result['em']}")
        print(f"🔄 Iterations Used: {result['iterations_used']}")
        print(f"⏱️  Search Time: {result['search_time']:.2f} seconds")
        print(f"🌳 Nodes Explored: {result['nodes_explored']}")
        print(f"🔤 Tokens Used: {result['tokens_used']}")
        print(f"🌐 API Calls: {result['api_calls']}")
        
        # Step 9: Show trajectory
        if result['trajectory']:
            print(f"\n📋 SOLUTION TRAJECTORY:")
            print("-" * 40)
            for i, step in enumerate(result['trajectory'], 1):
                if step.get('thought'):
                    print(f"Thought {i}: {step['thought']}")
                if step.get('action'):
                    print(f"Action {i}: {step['action']}")
                if step.get('observation'):
                    obs = step['observation']
                    if len(obs) > 200:
                        obs = obs[:200] + "..."
                    print(f"Observation {i}: {obs}")
                print()
        
        # Step 10: Save outputs if configured
        output_config = config.get('output', {})
        output_dir = Path(args.output_dir)
        
        if output_config.get('save_trajectories', False):
            traj_dir = output_dir / output_config.get('trajectories_dir', 'trajectories')
            traj_dir.mkdir(parents=True, exist_ok=True)
            
            traj_file = traj_dir / f"task_{args.task_index}.json"
            with open(traj_file, 'w') as f:
                json.dump({
                    'question_index': args.task_index,
                    'question': question,
                    'question_info': question_info,
                    'result': result
                }, f, indent=2)
            
            print(f"💾 Trajectory saved to {traj_file}")
            logger.info(f"Trajectory saved to {traj_file}")
        
        if output_config.get('save_tree_viz', False):
            try:
                viz_dir = output_dir / output_config.get('tree_viz_dir', 'visualizations')
                viz_dir.mkdir(parents=True, exist_ok=True)
                
                viz_file = viz_dir / f"task_{args.task_index}.png"
                lats.visualize_tree(result['search_tree'], str(viz_file))
                
                print(f"📊 Tree visualization saved to {viz_file}")
                logger.info(f"Tree visualization saved to {viz_file}")
            except Exception as e:
                logger.warning(f"Could not save tree visualization: {e}")
        
        # Step 11: Summary statistics
        if output_config.get('save_stats', False):
            stats_file = output_dir / output_config.get('stats_file', 'run_statistics.json')
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            
            stats = {
                'timestamp': result.get('timestamp', ''),
                'config': args.config,
                'task_index': args.task_index,
                'question_info': question_info,
                'performance': {
                    'is_correct': result['is_correct'],
                    'f1_score': result['score'],
                    'exact_match': result['em'],
                    'search_time': result['search_time'],
                    'iterations_used': result['iterations_used'],
                    'nodes_explored': result['nodes_explored'],
                    'tokens_used': result['tokens_used'],
                    'api_calls': result['api_calls']
                }
            }
            
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
            
            print(f"📈 Statistics saved to {stats_file}")
            logger.info(f"Statistics saved to {stats_file}")
        
        print("\n🏁 Example completed successfully!")
        logger.info("HotPotQA example completed successfully")
        
        # Return exit code based on correctness
        return 0 if result['is_correct'] else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        logger.info("Execution interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Execution failed: {e}", exc_info=True)
        return 1


def run_multiple_questions(config_path: str, start_idx: int = 0, end_idx: int = 10):
    """
    Run LATS on multiple questions for evaluation.
    
    Args:
        config_path: Path to configuration file
        start_idx: Starting question index
        end_idx: Ending question index (exclusive)
    """
    print(f"🔄 Running LATS on questions {start_idx} to {end_idx-1}")
    
    results = []
    total_correct = 0
    
    for i in range(start_idx, end_idx):
        print(f"\n{'='*20} Question {i} {'='*20}")
        
        # Run main function with current index
        sys.argv = ['hotpotqa_example.py', '--config', config_path, '--task-index', str(i)]
        exit_code = main()
        
        # Track results
        is_correct = (exit_code == 0)
        results.append(is_correct)
        if is_correct:
            total_correct += 1
        
        print(f"Question {i}: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
    
    # Summary
    accuracy = total_correct / len(results) if results else 0
    print(f"\n{'='*50}")
    print(f"📊 OVERALL RESULTS")
    print(f"{'='*50}")
    print(f"Questions processed: {len(results)}")
    print(f"Correct answers: {total_correct}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return results


if __name__ == "__main__":
    # Check for batch mode
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        config_path = sys.argv[2] if len(sys.argv) > 2 else "config/examples/hotpotqa_config.yaml"
        start_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        end_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        run_multiple_questions(config_path, start_idx, end_idx)
    else:
        exit_code = main()
        sys.exit(exit_code)