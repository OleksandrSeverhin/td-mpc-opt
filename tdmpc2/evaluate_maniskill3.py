import argparse
import os
import torch
import numpy as np

os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

from envs import make_env
from tdmpc2 import TDMPC2

def evaluate(cfg):
    """
    Evaluates a trained TD-MPC2 agent in a given environment.

    Args:
        cfg (object): A configuration object containing experiment parameters.
                      It must have attributes like `checkpoint_path`, `task_name`,
                      `seed`, `episode_length`, `num_eval_episodes`, `multitask`,
                      `pixels`, etc.
    """
    # 1. Device and Logging Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Agent Initialization
    # A dummy config object is created for the agent's initialization.
    # In a real setup, this would come from a loaded config file.
    if not hasattr(cfg, 'action_dim'):
        # This is a placeholder. In a real scenario, these values would be
        # loaded from the config file saved during training.
        cfg.action_dim = 6  # Example action dimension
        cfg.latent_dim = 50 # Example latent dimension
        cfg.discount_min = 0.5
        cfg.discount_max = 0.99
        cfg.discount_denom = 1000
        cfg.num_pi_trajs = 1
        cfg.num_samples = 64
        cfg.num_elites = 8
        cfg.min_std = 0.1
        cfg.max_std = 1.0
        cfg.temperature = 1.0
        cfg.horizon = 5
        cfg.iterations = 1
        cfg.grad_clip_norm = 10.0
        cfg.entropy_coef = 0.1
        cfg.consistency_coef = 1.0
        cfg.reward_coef = 1.0
        cfg.value_coef = 1.0
        cfg.num_q = 2
        cfg.lr = 1e-4
        cfg.enc_lr_scale = 1.0
        cfg.mpc = True


    agent = TDMPC2(cfg)
    print("TD-MPC2 agent initialized.")

    # 3. Load the pre-trained model
    if not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at: {cfg.checkpoint_path}")
    
    print(f"Loading model from {cfg.checkpoint_path}...")
    try:
        agent.load(cfg.checkpoint_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 4. Environment and Evaluation Loop
    # The evaluation script should handle either a single task or multiple tasks
    if cfg.multitask:
        tasks = cfg.task_names
    else:
        tasks = [cfg.task_name]

    all_rewards = []
    print("\nStarting evaluation...")

    for task_idx, task_name in enumerate(tasks):
        print(f"\n--- Evaluating task: {task_name} ---")
        
        # Create the environment for the current task
        env = make_env(cfg)
        
        episode_rewards = []

        for episode in range(cfg.num_eval_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            t0 = True

            for t in range(cfg.episode_length):
                # Get the action from the agent's policy in evaluation mode
                action = agent.act(obs, t0=t0, eval_mode=True, task=task_idx)
                
                # Step the environment
                obs, reward, done, _ = env.step(action.cpu().numpy())
                episode_reward += reward
                t0 = False

                if done:
                    break
            
            episode_rewards.append(episode_reward)
            print(f"Episode {episode+1}/{cfg.num_eval_episodes} - Reward: {episode_reward:.2f}")

        avg_reward = np.mean(episode_rewards)
        all_rewards.append(avg_reward)
        print(f"Task '{task_name}' Average Reward over {cfg.num_eval_episodes} episodes: {avg_reward:.2f}")

    # 5. Final results summary
    total_avg_reward = np.mean(all_rewards)
    print("\n=========================")
    print(f"Final Average Reward: {total_avg_reward:.2f}")
    print("=========================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TD-MPC2 Evaluation Script')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the saved model checkpoint file.')
    parser.add_argument('--task_name', type=str, default='walker_stand',
                        help='Name of the environment task to evaluate.')
    parser.add_argument('--num_eval_episodes', type=int, default=10,
                        help='Number of episodes to run for evaluation.')
    
    # Assuming other necessary config parameters would be parsed here
    parser.add_argument('--multitask', action='store_true',
                        help='Enable for multi-task evaluation.')
    parser.add_argument('--task_names', nargs='+', default=['walker_stand', 'cheetah_run'],
                        help='List of tasks for multi-task evaluation.')
    parser.add_argument('--episode_length', type=int, default=1000,
                        help='Length of each episode.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--pixels', action='store_true',
                        help='Use pixel observations instead of state.')

    args = parser.parse_args()

    # Create a simple config object from the parsed arguments
    class Config:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    cfg = Config(**vars(args))

    evaluate(cfg)
