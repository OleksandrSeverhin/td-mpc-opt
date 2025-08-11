import os
import time
from datetime import timedelta
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import wandb
import hydra
import imageio
import numpy as np
import pandas as pd
import torch
from termcolor import colored
from tqdm import tqdm
from pathlib import Path
import traceback
import json

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from omegaconf import OmegaConf
from tdmpc2 import TDMPC2

# Try to import ManiSkill
try:
    import gymnasium as gym
    MANISKILL_AVAILABLE = True
except ImportError:
    print(colored("ManiSkill not found. Install with: pip install mani_skill", 'yellow'))
    MANISKILL_AVAILABLE = False

torch.backends.cudnn.benchmark = True

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def create_maniskill_env(task_name: str, cfg):
    """Create a ManiSkill 3 environment"""
    if not MANISKILL_AVAILABLE:
        raise ImportError("ManiSkill not available")
    
    # Get task-specific config or use defaults
    task_config = cfg.maniskill3.task_configs.get(task_name, {}) if hasattr(cfg, 'maniskill3') else {}
    env_kwargs = dict(cfg.maniskill3.env_kwargs) if hasattr(cfg, 'maniskill3') else {}
    env_kwargs.update(task_config)
    
    try:
        env = gym.make(task_name, **env_kwargs)
        return env
    except Exception as e:
        print(colored(f"Failed to create environment {task_name}: {str(e)}", 'red'))
        raise

def evaluate_maniskill_task(agent, task_name: str, cfg) -> dict:
    """Evaluate agent on a single ManiSkill 3 task"""
    print(colored(f"Evaluating ManiSkill 3 task: {task_name}", 'cyan', attrs=['bold']))
    
    try:
        # Create environment
        env = create_maniskill_env(task_name, cfg)
        
        # Get number of episodes for this task
        task_config = cfg.maniskill3.task_configs.get(task_name, {}) if hasattr(cfg, 'maniskill3') else {}
        num_episodes = task_config.get('eval_episodes', cfg.eval_episodes)
        
        # Prepare video directory if needed
        video_dir = None
        if cfg.save_video:
            video_dir = os.path.join(cfg.work_dir, 'videos', 'maniskill3', task_name)
            os.makedirs(video_dir, exist_ok=True)
        
        # Run evaluation episodes
        ep_rewards = []
        ep_successes = []
        ep_lengths = []
        
        for episode in range(num_episodes):
            try:
                obs, info = env.reset(seed=cfg.seed + episode)
                done = False
                ep_reward = 0
                ep_length = 0
                frames = []
                
                # Get max episode steps
                max_steps = task_config.get('max_episode_steps', 200)
                if cfg.save_video:
                    frames = [env.render()]
                
                while not done and ep_length < max_steps:
                    # Get action from agent
                    action = agent.act(obs, t0=(ep_length==0))
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    ep_reward += reward
                    ep_length += 1
                    done = terminated or truncated
                    
                    if cfg.save_video:
                        frames.append(env.render())
                
                # Record episode results
                ep_rewards.append(ep_reward)
                ep_successes.append(info.get('success', False))
                ep_lengths.append(ep_length)
                
                # Save video if requested
                if cfg.save_video and frames:
                    video_path = os.path.join(video_dir, f'episode_{episode+1}.mp4')
                    imageio.mimsave(video_path, frames, fps=15)
                
                success_str = "Success" if info.get('success', False) else "Failure"
                print(f"    Episode {episode+1:2d}/{num_episodes}: "
                      f"R={ep_reward:7.3f}, L={ep_length:3d}, {success_str}")
                
            except Exception as e:
                print(colored(f"Episode {episode+1} failed: {str(e)}", 'red'))
                ep_rewards.append(0.0)
                ep_successes.append(False)
                ep_lengths.append(0)
        
        # Calculate statistics
        avg_reward = np.mean(ep_rewards)
        success_rate = np.mean(ep_successes)
        avg_length = np.mean(ep_lengths)
        
        results = {
            'task_name': task_name,
            'num_episodes': len(ep_rewards),
            'avg_reward': avg_reward,
            'std_reward': np.std(ep_rewards),
            'min_reward': np.min(ep_rewards),
            'max_reward': np.max(ep_rewards),
            'success_rate': success_rate,
            'avg_episode_length': avg_length,
            'std_episode_length': np.std(ep_lengths),
            'successful_episodes': sum(ep_successes),
            'failed_episodes': len(ep_successes) - sum(ep_successes),
            'all_rewards': ep_rewards,
            'all_successes': ep_successes,
            'all_lengths': ep_lengths
        }
        
        print(colored(f"{task_name:<22} "
                     f"R: {avg_reward:.3f}±{results['std_reward']:.3f}  "
                     f"S: {success_rate:.3f}  "
                     f"L: {avg_length:.1f}", 'yellow'))
        
        # Log to W&B
        wandb.log({
            f"maniskill3/{task_name}/avg_reward": avg_reward,
            f"maniskill3/{task_name}/success_rate": success_rate,
            f"maniskill3/{task_name}/avg_length": avg_length,
        })
        
        env.close()
        return results
        
    except Exception as e:
        print(colored(f"Task {task_name} failed completely: {str(e)}", 'red'))
        return {
            'task_name': task_name,
            'num_episodes': 0,
            'error': str(e),
            'avg_reward': 0.0,
            'std_reward': 0.0,
            'success_rate': 0.0,
            'avg_episode_length': 0.0,
            'successful_episodes': 0,
            'failed_episodes': cfg.eval_episodes
        }

def evaluate_multitask_maniskill(agent, cfg):
    """Evaluate agent on all ManiSkill 3 tasks"""
    if not hasattr(cfg, 'maniskill3') or not hasattr(cfg.maniskill3, 'all_tasks'):
        print(colored("No ManiSkill 3 tasks specified in config", 'red'))
        return {}, []
    
    tasks = cfg.maniskill3.all_tasks
    total_tasks = len(tasks)
    
    print(colored(f"Starting ManiSkill 3 multi-task evaluation on {total_tasks} tasks", 'green', attrs=['bold']))
    print(colored(f"Results directory: {cfg.work_dir}", 'blue'))
    print(colored(f"Using checkpoint: {cfg.checkpoint}", 'blue'))
    print("="*90)
    
    all_results = {}
    summary_results = []
    failed_tasks = []
    
    start_time = time.time()
    
    for i, task_name in enumerate(tasks, 1):
        print(colored(f"[{i:2d}/{total_tasks}] {task_name}", 'magenta', attrs=['bold']))
        
        # Evaluate single task
        task_result = evaluate_maniskill_task(agent, task_name, cfg)
        all_results[task_name] = task_result
        
        # Add to summary
        summary_row = {
            'task': task_name,
            'avg_reward': task_result['avg_reward'],
            'std_reward': task_result['std_reward'],
            'success_rate': task_result['success_rate'],
            'avg_length': task_result['avg_episode_length'],
            'successful_episodes': task_result['successful_episodes'],
            'failed_episodes': task_result['failed_episodes']
        }
        
        if 'error' in task_result:
            summary_row['error'] = task_result['error']
            failed_tasks.append(task_name)
        
        summary_results.append(summary_row)
        
        # Continue on failure if specified
        if cfg.get('continue_on_failure', True) and 'error' in task_result:
            print(colored(f"Continuing despite failure in {task_name}", 'yellow'))
    
    # Print summary
    total_time = time.time() - start_time
    print("\n" + "="*90)
    print(colored("MANISKILL 3 EVALUATION SUMMARY", 'green', attrs=['bold']))
    print("="*90)
    
    df = pd.DataFrame(summary_results)
    print(df.to_string(index=False, float_format='%.3f'))
    
    # Calculate overall metrics
    successful_tasks = df[df['success_rate'] > 0]
    overall_success_rate = df['success_rate'].mean()
    overall_avg_reward = df['avg_reward'].mean()
    
    print(colored(f"Overall Performance:", 'cyan', attrs=['bold']))
    print(colored(f"   Tasks with success > 0: {len(successful_tasks)}/{len(df)}", 'green'))
    print(colored(f"   Average success rate: {overall_success_rate:.3f}", 'green'))
    print(colored(f"   Average reward: {overall_avg_reward:.3f}", 'green'))
    print(colored(f"   Failed tasks: {len(failed_tasks)}", 'red'))
    print(colored(f"   Total evaluation time: {format_time(total_time)}", 'blue'))
    
    if failed_tasks:
        print(colored(f"   Failed task list: {', '.join(failed_tasks)}", 'red'))
    
    # Save results
    if cfg.save_csv:
        results_file = os.path.join(cfg.work_dir, "maniskill3_evaluation_results.csv")
        df.to_csv(results_file, index=False)
        print(colored(f"\Results saved to: {results_file}", 'blue'))
        
        # Save detailed results as JSON
        detailed_file = os.path.join(cfg.work_dir, "maniskill3_detailed_results.json")
        with open(detailed_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(colored(f"Detailed results saved to: {detailed_file}", 'blue'))
    
    # Log overall metrics to W&B
    wandb.log({
        "maniskill3/overall/success_rate": overall_success_rate,
        "maniskill3/overall/avg_reward": overall_avg_reward,
        "maniskill3/overall/successful_tasks": len(successful_tasks),
        "maniskill3/overall/failed_tasks": len(failed_tasks),
        "maniskill3/overall/total_tasks": len(df),
        "maniskill3/overall/evaluation_time": total_time
    })
    
    return all_results, df

@hydra.main(config_name='config_maniskill3', config_path='./student_config')
def evaluate(cfg: dict):
	"""
	Enhanced script for evaluating TD-MPC2 checkpoint on ManiSkill 3 tasks.

	Key improvements:
	- Multi-task evaluation on all ManiSkill 3 tasks
	- Robust error handling with continue-on-failure
	- Detailed logging and result saving
	- W&B integration for comprehensive tracking
	- Video saving for successful episodes
	
	Example usage:
	```
		$ python evaluate.py  # Evaluate on all ManiSkill 3 tasks
		$ python evaluate.py eval_episodes=5  # Fewer episodes per task
		$ python evaluate.py save_video=true  # Save videos
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.eval_episodes > 0, 'Must evaluate at least 1 episode.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)

	# Initialize W&B
	wandb.init(
        project=cfg.get('wandb_project', 'td-mpc-maniskill3'),
        config=OmegaConf.to_container(cfg, resolve=True),
        name=f"maniskill3_eval_{cfg.get('model_size', 'student')}M",
        tags=['maniskill3', 'evaluation', 'multitask'],
        group="maniskill3_evaluation"
    )

	print(colored(f'Task: ManiSkill 3 Multi-Task Evaluation', 'blue', attrs=['bold']))
	print(colored(f'Model: Student Model (distilled)', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	print(colored(f'Episodes per task: {cfg.eval_episodes}', 'blue', attrs=['bold']))

	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found! Must be a valid filepath.'
	agent.load(cfg.checkpoint)
	print(colored('Student model loaded successfully', 'green', attrs=['bold']))

	# Check if ManiSkill is available
	if not MANISKILL_AVAILABLE:
		print(colored('ManiSkill not available. Please install with: pip install mani_skill', 'red', attrs=['bold']))
		return

	# Create results directory
	os.makedirs(cfg.work_dir, exist_ok=True)

	# Run ManiSkill 3 multi-task evaluation
	print(colored('Starting ManiSkill 3 evaluation...', 'cyan', attrs=['bold']))
	all_results, summary_df = evaluate_multitask_maniskill(agent, cfg)

	# Final summary
	if len(summary_df) > 0:
		successful_tasks = len(summary_df[summary_df['success_rate'] > 0])
		total_tasks = len(summary_df)
		overall_success = summary_df['success_rate'].mean()
		
		print(colored(f'Final Results:', 'green', attrs=['bold']))
		print(colored(f'   Successful tasks: {successful_tasks}/{total_tasks} ({successful_tasks/total_tasks:.1%})', 'green'))
		print(colored(f'   Overall success rate: {overall_success:.3f}', 'green'))
		print(colored(f'   Average reward: {summary_df["avg_reward"].mean():.3f}', 'green'))
		
		# Log final summary to W&B
		wandb.log({
			"final/successful_task_ratio": successful_tasks / total_tasks,
			"final/overall_success_rate": overall_success,
			"final/total_tasks_evaluated": total_tasks
		})

if __name__ == '__main__':
	evaluate()
	wandb.finish()