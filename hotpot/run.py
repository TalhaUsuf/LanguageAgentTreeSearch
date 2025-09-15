import os
import json
import argparse
from datetime import datetime

from hotpotqa import HotPotQATask
from models import gpt_usage
from lats import lats_search
from tot import dfs_search
from rap import mcts_search
from logger_config import setup_logger, get_context_logger
from trajectory_saver import trajectory_saver

def run(args):
    # Setup enhanced logging
    setup_logger(args.log, debug_mode=True)
    ctx_logger = get_context_logger()
    
    task = HotPotQATask()
    ctx_logger.info(f"🚀 Starting LATS experiment with {task}")
    ctx_logger.info(f"Algorithm: {args.algorithm}, Backend: {args.backend}")
    ctx_logger.info(f"Task range: {args.task_start_index}-{args.task_end_index}")
    ctx_logger.info(f"Parameters: iterations={args.iterations}, n_generate={args.n_generate_sample}, n_evaluate={args.n_evaluate_sample}")
    
    # create log and trajectory directories if they don't exist
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    os.makedirs("trajs", exist_ok=True)

    task_accs = []

    for i in range(args.task_start_index, args.task_end_index):
        ctx_logger.info(f"\n{'='*60}")
        ctx_logger.info(f"🔍 Processing Question {i+1}/{args.task_end_index} (index {i})")
        ctx_logger.info(f"{'='*60}")
        
        # solve
        start_time = datetime.now()
        if args.algorithm == 'lats':
            _, _, _, reward, em = lats_search(args, task, i, args.iterations, True)
        elif args.algorithm == 'tot':
            _, _, _, reward, em = dfs_search(args, task, i, args.iterations)
        elif args.algorithm == 'rap':
            _, _, _, reward, em = mcts_search(args, task, i, args.iterations)
        else:
            raise Exception("Search algorithm option not valid")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # log main metric
        if em is None:
            em = 0
        task_accs.append(em)
        cnt_avg = sum(task_accs) / len(task_accs)
        
        success_emoji = "✅" if reward > 0 else "❌"
        ctx_logger.info(f"{success_emoji} Question {i}: EM={em:.3f}, Reward={reward:.3f}, Duration={duration:.1f}s")
        ctx_logger.info(f"📊 Running Average: {cnt_avg:.3f} ({sum(task_accs)}/{len(task_accs)} correct)")
        
        print(f"Question {i}: EM={em:.3f}, Avg={cnt_avg:.3f}, Time={duration:.1f}s")
        #all_nodes_dict = [(node.to_dict(), value) for node, value in all_nodes]
        
       
    # Final results and cleanup
    n = args.task_end_index - args.task_start_index
    final_accuracy = sum(task_accs) / len(task_accs) if task_accs else 0.0
    successful_tasks = sum(1 for acc in task_accs if acc > 0)
    
    ctx_logger.info(f"\n{'='*60}")
    ctx_logger.info(f"🏁 EXPERIMENT COMPLETED")
    ctx_logger.info(f"{'='*60}")
    ctx_logger.info(f"📈 Final Results:")
    ctx_logger.info(f"   Total Questions: {n}")
    ctx_logger.info(f"   Successful: {successful_tasks}")
    ctx_logger.info(f"   Success Rate: {successful_tasks/n*100:.1f}%")
    ctx_logger.info(f"   Average EM: {final_accuracy:.3f}")
    ctx_logger.info(f"   LLM Usage: {gpt_usage(args.backend)}")
    
    # Save all trajectories
    trajectory_file = trajectory_saver.save_trajectories()
    training_file = trajectory_saver.save_training_format()
    
    ctx_logger.info(f"💾 Saved trajectories to: {trajectory_file}")
    ctx_logger.info(f"🎯 Saved training data to: {training_file}")
    
    print(f"\n🏁 Final Results: {successful_tasks}/{n} success ({successful_tasks/n*100:.1f}%), EM={final_accuracy:.3f}")
    print(f"💾 Trajectories saved to: {trajectory_file}")
    print(f"🎯 Training data saved to: {training_file}")
    print(f"📊 LLM Usage: {gpt_usage(args.backend)}")

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, choices=['gpt-4', 'gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'llama-3.1-70b'], default='llama-3.1-70b')
    args.add_argument('--temperature', type=float, default=1.0)
    args.add_argument('--task_start_index', type=int, default=900)
    args.add_argument('--task_end_index', type=int, default=1000)
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'])
    args.add_argument('--n_generate_sample', type=int, default=1)  
    args.add_argument('--n_evaluate_sample', type=int, default=1)
    args.add_argument('--iterations', type=int, default=50)
    args.add_argument('--log', type=str)
    args.add_argument('--algorithm', type=str, choices=['lats', 'rap', 'tot'])

    args = args.parse_args()
    return args

if __name__ == '__main__':
    import pdb
    import signal
    
    def signal_handler(sig, frame):
        print(f"\n🛑 Interrupted by user (Ctrl+C)")
        pdb.set_trace()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        args = parse_args()
        print(f"🔧 Configuration: {args}")
        run(args)
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        # pdb.post_mortem() starts the debugger in post-mortem mode, allowing you to
        # inspect the state of the program at the point where the exception occurred.
        # You can examine variables, call stack, and debug the issue that caused the crash.
        pdb.post_mortem()