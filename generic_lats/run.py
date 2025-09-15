"""
Main runner for Generic LATS Framework

This script provides a unified entry point for running LATS on different tasks
with flexible configuration options.

Usage:
    python run.py --config config/examples/hotpotqa_config.yaml
    python run.py --task hotpotqa --start-index 0 --end-index 10
    python run.py --config custom_config.yaml --verbose
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Any

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from core.lats_algorithm import LATSSearch
from tasks.hotpotqa_task import HotPotQATask
from tasks.base_task import BaseTask
from tools.tool_registry import ToolRegistry
from tools.wikipedia_tools import WikipediaSearchTool, WikipediaLookupTool
from tools.base_tools import FinishTool
from models.llm_wrapper import LangChainLLM
from utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class TaskFactory:
    """Factory for creating task instances based on configuration."""
    
    @staticmethod
    def create_task(task_config: Dict[str, Any], **kwargs) -> BaseTask:
        """
        Create a task instance based on configuration.
        
        Args:
            task_config: Task configuration dictionary
            **kwargs: Additional arguments for task initialization
            
        Returns:
            Task instance
        """
        task_type = task_config.get('type', 'hotpotqa')
        
        if task_type == 'hotpotqa':
            dataset_config = task_config.get('dataset', {})
            return HotPotQATask(
                dataset_path=dataset_config.get('path', 'data/hotpot_dev_v1_simplified.json'),
                split=dataset_config.get('split', 'dev'),
                **kwargs
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")


def setup_logging(config: Dict[str, Any], verbose: bool = False):
    """Setup logging based on configuration."""
    log_config = config.get('logging', {})
    
    # Override with verbose if specified
    level_name = 'DEBUG' if verbose else log_config.get('level', 'INFO')
    level = getattr(logging, level_name.upper())
    
    # Create logs directory
    log_file = log_config.get('file', 'logs/lats_run.log')
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    formatter = logging.Formatter(
        log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.setLevel(level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_config = log_config.get('console', {})
    if console_config.get('enabled', True):
        console_handler = logging.StreamHandler()
        console_level = getattr(logging, console_config.get('level', level_name).upper())
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


def setup_tools(config: Dict[str, Any]) -> ToolRegistry:
    """Setup and register tools based on configuration."""
    tool_config = config.get('tools', {})
    enabled_tools = tool_config.get('enabled_tools', ['search', 'lookup', 'finish'])
    
    tool_registry = ToolRegistry()
    
    # Register standard tools
    if 'search' in enabled_tools:
        tool_registry.register('search', WikipediaSearchTool())
    
    if 'lookup' in enabled_tools:
        tool_registry.register('lookup', WikipediaLookupTool())
    
    if 'finish' in enabled_tools:
        tool_registry.register('finish', FinishTool())
    
    # Load custom tools from directory if specified
    custom_tools_dir = tool_config.get('custom_tools_dir')
    if custom_tools_dir and Path(custom_tools_dir).exists():
        tool_registry.load_from_directory(custom_tools_dir)
    
    logger.info(f"Registered {len(tool_registry)} tools: {list(tool_registry.get_tool_names())}")
    
    return tool_registry


def run_single_question(config: Dict[str, Any], question_idx: int) -> Dict[str, Any]:
    """
    Run LATS on a single question.
    
    Args:
        config: Configuration dictionary
        question_idx: Index of question to solve
        
    Returns:
        Result dictionary
    """
    # Initialize components
    llm = LangChainLLM(config['llm'])
    tool_registry = setup_tools(config)
    
    # Create task
    task = TaskFactory.create_task(
        config['task'],
        tool_registry=tool_registry,
        llm=llm,
        config=config
    )
    
    # Validate question index
    if question_idx >= len(task):
        raise IndexError(f"Question index {question_idx} out of range (0-{len(task)-1})")
    
    # Initialize LATS
    lats = LATSSearch(task, llm, tool_registry, config['algorithm'])
    
    # Get question and run search
    question = task.get_input(question_idx)
    logger.info(f"Solving question {question_idx}: {question[:100]}...")
    
    result = lats.search(question, question_idx)
    
    return result


def run_multiple_questions(config: Dict[str, Any], 
                          start_idx: int = 0, 
                          end_idx: int = None) -> List[Dict[str, Any]]:
    """
    Run LATS on multiple questions.
    
    Args:
        config: Configuration dictionary
        start_idx: Starting question index
        end_idx: Ending question index (exclusive)
        
    Returns:
        List of result dictionaries
    """
    # Initialize components
    llm = LangChainLLM(config['llm'])
    tool_registry = setup_tools(config)
    
    # Create task
    task = TaskFactory.create_task(
        config['task'],
        tool_registry=tool_registry,
        llm=llm,
        config=config
    )
    
    # Determine range
    if end_idx is None:
        end_idx = len(task)
    
    end_idx = min(end_idx, len(task))
    
    if start_idx >= end_idx:
        raise ValueError(f"Invalid range: start_idx={start_idx}, end_idx={end_idx}")
    
    logger.info(f"Running LATS on questions {start_idx} to {end_idx-1}")
    
    # Initialize LATS
    lats = LATSSearch(task, llm, tool_registry, config['algorithm'])
    
    results = []
    
    for idx in range(start_idx, end_idx):
        try:
            question = task.get_input(idx)
            logger.info(f"Processing question {idx}: {question[:100]}...")
            
            result = lats.search(question, idx)
            results.append(result)
            
            # Log progress
            success = "✅" if result['is_correct'] else "❌"
            logger.info(f"Question {idx}: {success} Score={result['score']:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing question {idx}: {e}")
            # Add error result
            results.append({
                'question_index': idx,
                'error': str(e),
                'is_correct': False,
                'score': 0.0
            })
    
    return results


def save_results(results: List[Dict[str, Any]], output_config: Dict[str, Any], 
                args: argparse.Namespace):
    """Save results to various output formats."""
    if not output_config.get('save_trajectories', False):
        return
    
    # Create output directory
    output_dir = Path(output_config.get('trajectories_dir', 'output/trajectories'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual results
    for result in results:
        if 'question_index' in result:
            idx = result['question_index']
            result_file = output_dir / f"question_{idx}.json"
            
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
    
    # Save summary
    summary_file = output_dir / "summary.json"
    
    summary = {
        'run_info': {
            'config_file': getattr(args, 'config', 'default'),
            'start_index': getattr(args, 'start_index', 0),
            'end_index': getattr(args, 'end_index', len(results)),
            'total_questions': len(results),
            'timestamp': time.time()
        },
        'performance': {
            'total_correct': sum(1 for r in results if r.get('is_correct', False)),
            'accuracy': sum(1 for r in results if r.get('is_correct', False)) / len(results) if results else 0,
            'average_score': sum(r.get('score', 0) for r in results) / len(results) if results else 0,
            'average_time': sum(r.get('search_time', 0) for r in results) / len(results) if results else 0,
            'total_tokens': sum(r.get('tokens_used', 0) for r in results),
            'total_api_calls': sum(r.get('api_calls', 0) for r in results)
        },
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generic LATS Framework Runner")
    
    # Configuration
    parser.add_argument(
        '--config', 
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    # Task selection
    parser.add_argument(
        '--task',
        type=str,
        choices=['hotpotqa'],
        help='Task type (overrides config)'
    )
    
    # Question range
    parser.add_argument(
        '--start-index',
        type=int,
        default=0,
        help='Starting question index'
    )
    
    parser.add_argument(
        '--end-index',
        type=int,
        help='Ending question index (exclusive)'
    )
    
    parser.add_argument(
        '--single',
        type=int,
        help='Run on a single question index'
    )
    
    # Algorithm parameters
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of LATS iterations (overrides config)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        help='LLM temperature (overrides config)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # Utility options
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration file and exit'
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = ConfigLoader.load(args.config)
        
        # Validate configuration
        if args.validate_config:
            print("✅ Configuration is valid")
            return 0
        
        # Apply command line overrides
        if args.task:
            config['task']['type'] = args.task
        
        if args.iterations:
            config['algorithm']['iterations'] = args.iterations
        
        if args.temperature:
            config['llm']['temperature'] = args.temperature
        
        if args.end_index is None and 'task' in config and 'dataset' in config['task']:
            args.end_index = config['task']['dataset'].get('end_index')
        
        # Setup logging
        setup_logging(config, args.verbose)
        logger.info("Starting Generic LATS Framework")
        
        # Run LATS
        if args.single is not None:
            print(f"Running LATS on question {args.single}")
            result = run_single_question(config, args.single)
            results = [result]
            
            # Display result
            print(f"\n{'='*50}")
            print(f"Result: {'✅ CORRECT' if result['is_correct'] else '❌ INCORRECT'}")
            print(f"Answer: {result['answer']}")
            print(f"Ground Truth: {result['ground_truth']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Time: {result['search_time']:.2f}s")
            
        else:
            print(f"Running LATS on questions {args.start_index} to {args.end_index or 'end'}")
            results = run_multiple_questions(config, args.start_index, args.end_index)
            
            # Display summary
            total_correct = sum(1 for r in results if r.get('is_correct', False))
            accuracy = total_correct / len(results) if results else 0
            avg_score = sum(r.get('score', 0) for r in results) / len(results) if results else 0
            
            print(f"\n{'='*50}")
            print(f"SUMMARY RESULTS")
            print(f"{'='*50}")
            print(f"Questions processed: {len(results)}")
            print(f"Correct answers: {total_correct}")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Average score: {avg_score:.4f}")
        
        # Save results
        output_config = config.get('output', {})
        if output_config:
            # Update output directory if specified
            if args.output_dir != 'output':
                output_config['trajectories_dir'] = f"{args.output_dir}/trajectories"
            
            save_results(results, output_config, args)
        
        logger.info("Generic LATS Framework completed successfully")
        
        # Return success if all questions were correct (for single question mode)
        if args.single is not None:
            return 0 if results[0].get('is_correct', False) else 1
        else:
            return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        logger.info("Execution interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Execution failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)