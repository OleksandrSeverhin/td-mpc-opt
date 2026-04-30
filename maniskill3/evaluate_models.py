import os
import torch
import numpy as np
import gymnasium as gym
import mani_skill.envs  # Crucial: registers ManiSkill3 environments
from omegaconf import OmegaConf
import pandas as pd
from pathlib import Path

from tdmpc2 import TDMPC2

# --- THESIS EVALUATION SETTINGS ---
EVAL_EPISODES = 12500
LOG_DIR = Path("logs")

def get_eval_cfg():
    cfg = OmegaConf.load("config.yaml")
    OmegaConf.set_struct(cfg, False)
    
    cfg.action_dim = 6
    cfg.obs_shape = {'state': [24]}
    cfg.model_size = 1
    cfg.num_enc_layers = 2
    cfg.enc_dim = 256
    cfg.mlp_dim = 512
    cfg.latent_dim = 512
    cfg.horizon = 5
    cfg.seed = 42
    cfg.episode_length = 100 
    cfg.multitask = False
    cfg.task_dim = 0 
    
    return cfg

def evaluate_checkpoint(model_path, env_id):
    cfg = get_eval_cfg()
    
    # Initialize Student
    agent = TDMPC2(cfg)
    agent.load(model_path)
    agent.model.eval()
    
    # Initialize Live Physics Environment
    env = gym.make(env_id, obs_mode="state")
    env_act_dim = env.action_space.shape[-1]
    
    successes = 0
    returns = []

    for ep in range(EVAL_EPISODES):
        reset_result = env.reset(seed=42+ep)
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        done = False
        ep_reward = 0
        
        while not done:
            # 1. Bulletproof Tensor Handling
            if not isinstance(obs, torch.Tensor):
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cuda')
            else:
                obs_tensor = obs.clone().detach().to(dtype=torch.float32, device='cuda')
                
            # 2. Slice strictly the LAST dimension (fixes the 1x35 matrix crash)
            obs_tensor = obs_tensor[..., :24]
            
            # 3. Add batch dimension if missing -> (1, 24)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # 4. Get the action from the Student
            with torch.no_grad():
                task_idx = torch.zeros(obs_tensor.shape[0], dtype=torch.long, device='cuda')
                z = agent.model.encode(obs_tensor, task=task_idx)
                action = agent.model.pi(z, task=task_idx)[0]
                
            # 5. Handle Action Padding dynamically on the GPU
            if action.shape[-1] < env_act_dim:
                pad_len = env_act_dim - action.shape[-1]
                env_action = torch.nn.functional.pad(action, (0, pad_len))
            else:
                env_action = action[..., :env_act_dim]
                
            # 6. Step the environment (ManiSkill3 accepts GPU tensors natively)
            step_result = env.step(env_action)
            
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                
                # Safely parse batched booleans
                is_term = terminated.item() if isinstance(terminated, torch.Tensor) and terminated.numel() == 1 else bool(np.all(terminated))
                is_trunc = truncated.item() if isinstance(truncated, torch.Tensor) and truncated.numel() == 1 else bool(np.all(truncated))
                done = is_term or is_trunc
            else:
                obs, reward, is_done, info = step_result
                done = is_done.item() if isinstance(is_done, torch.Tensor) and is_done.numel() == 1 else bool(np.all(is_done))
                
            # Safely log reward
            ep_reward += reward.item() if isinstance(reward, torch.Tensor) else float(np.sum(reward))
            
            if done:
                # Safely extract success flag
                if isinstance(info, dict) and 'success' in info:
                    succ = info['success']
                    succ_val = succ.item() if isinstance(succ, torch.Tensor) and succ.numel() == 1 else bool(np.all(succ))
                    if succ_val:
                        successes += 1
                returns.append(ep_reward)

    env.close()
    
    success_rate = (successes / EVAL_EPISODES) * 100
    avg_return = np.mean(returns)
    
    print(f"--> Success Rate: {success_rate:.1f}% | Avg Return: {avg_return:.2f}")
    return success_rate, avg_return

def main():
    if not LOG_DIR.exists():
        print(f"Error: {LOG_DIR} does not exist yet. Wait for the training script to save a model.")
        return
        
    results = []
    
    print("===========================================")
    print(" STARTING THESIS DATA HARVESTER")
    print("===========================================")
    
    # Crawl the logs directory
    for exp_dir in LOG_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
            
        # Parse the experiment parameters from the folder name
        parts = exp_dir.name.split('_')
        if len(parts) < 3:
            continue
            
        schedule = parts[1]
        env_id = parts[2]
        
        # Grab the newest model file in the folder (e.g. model_50000.pt or model.pt)
        models = list(exp_dir.glob("model*.pt"))
        if not models:
            print(f"Skipping {exp_dir.name} - No models saved yet.")
            continue
            
        models.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        target_model = models[0]
        
        print(f"\nEvaluating: {exp_dir.name}")
        print(f"Checkpoint: {target_model.name}")
        
        try:
            success_rate, avg_return = evaluate_checkpoint(str(target_model), env_id)
            
            results.append({
                'Task': env_id,
                'Decay Schedule': schedule.capitalize(),
                'Checkpoint': target_model.name,
                'Success Rate (%)': f"{success_rate:.1f}%",
                'Avg Reward': f"{avg_return:.2f}"
            })
            
            # Save incrementally so you never lose data
            df = pd.DataFrame(results)
            df = df.sort_values(by=['Task', 'Decay Schedule'])
            df.to_csv("thesis_results.csv", index=False)
            print(f"Saved {exp_dir.name} to thesis_results.csv")
            
        except Exception as e:
            print(f"Failed to evaluate {exp_dir.name}: {e}")

    print("\n===========================================")
    print(" EVALUATION COMPLETE.")
    print(" Check 'thesis_results.csv' for your table data!")
    print("===========================================")

if __name__ == "__main__":
    main()