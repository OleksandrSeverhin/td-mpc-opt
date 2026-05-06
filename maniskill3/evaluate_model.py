import os
from pathlib import Path

import gymnasium as gym
import mani_skill.envs
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from tdmpc2 import TDMPC2

EVAL_EPISODES = 12500
LOG_DIR = Path("logs")
CHECKPOINT_PATH = "logs/PushCube-v1/1/online_control_PushCube-v1_1M/models/final.pt"

def get_eval_cfg():
    cfg = OmegaConf.load("config.yaml")
    OmegaConf.set_struct(cfg, False)
    cfg.update({
        "action_dim": 7,
        "obs_shape": {"state": [35]},
        "model_size": -1,
        "num_enc_layers": 2,
        "enc_dim": 256,
        "mlp_dim": 384,
        "latent_dim": 128,
        "horizon": 5,
        "seed": 42,
        "episode_length": 50,
        "multitask": False,
        "task_dim": 0,
        "env_id": "None"
    })
    return cfg

def to_tensor(obs):
    if not isinstance(obs, torch.Tensor):
        return torch.tensor(obs, dtype=torch.float32, device="cuda")
    return obs.clone().detach().to(dtype=torch.float32, device="cuda")

def prepare_obs(obs):
    obs_tensor = to_tensor(obs)[..., :35]
    if obs_tensor.dim() == 1:
        obs_tensor = obs_tensor.unsqueeze(0)
    return obs_tensor

def pad_action(action, env_act_dim):
    if action.shape[-1] < env_act_dim:
        pad_len = env_act_dim - action.shape[-1]
        return torch.nn.functional.pad(action, (0, pad_len))
    return action[..., :env_act_dim]

def check_bool(val):
    if isinstance(val, torch.Tensor) and val.numel() == 1:
        return val.item()
    return bool(np.all(val))

def parse_step(step_result):
    if len(step_result) == 5:
        obs, reward, term, trunc, info = step_result
        done = check_bool(term) or check_bool(trunc)
    else:
        obs, reward, is_done, info = step_result
        done = check_bool(is_done)
    return obs, reward, done, info

def get_reward_val(reward):
    if isinstance(reward, torch.Tensor):
        return reward.item()
    return float(np.sum(reward))

def check_success(info):
    if isinstance(info, dict) and "success" in info:
        return check_bool(info["success"])
    return False

def run_eval_episodes(env, agent, env_act_dim):
    successes = 0
    returns = []

    for ep in range(EVAL_EPISODES):
        reset_res = env.reset(seed=42 + ep)
        obs = reset_res[0] if isinstance(reset_res, tuple) else reset_res

        done = False
        ep_reward = 0.0

        while not done:
            obs_tensor = prepare_obs(obs)
            task_idx = torch.zeros(
                obs_tensor.shape[0], dtype=torch.long, device="cuda"
            )

            with torch.no_grad():
                z = agent.model.encode(obs_tensor, task=task_idx)
                action = agent.model.pi(z, task=task_idx)[0]

            env_action = pad_action(action, env_act_dim)
            obs, reward, done, info = parse_step(env.step(env_action))
            ep_reward += get_reward_val(reward)

            if done:
                if check_success(info):
                    successes += 1
                returns.append(ep_reward)

    return successes, returns

def evaluate_checkpoint(model_path, env_id):
    cfg = get_eval_cfg()
    
    agent = TDMPC2(cfg)
    sd = torch.load(model_path, map_location="cuda")
    if "model" in sd:
        sd = sd["model"]
    agent.model.load_state_dict(sd)
    agent.model.eval()

    env = gym.make(env_id, obs_mode="state")
    env_act_dim = env.action_space.shape[-1]
    
    successes, returns = run_eval_episodes(env, agent, env_act_dim)
    env.close()

    success_rate = (successes / EVAL_EPISODES) * 100
    avg_return = np.mean(returns)

    print(
        f"-> Success Rate: {success_rate:.1f}% | "
        f"Avg Return: {avg_return:.2f}"
    )
    return success_rate, avg_return

def main():
    results = []
    target_model = Path(CHECKPOINT_PATH)
    env_id = "PushCube-v1"
    
    if not target_model.exists():
        print(f"Error: {target_model} missing.")
        return

    print(f"Evaluating: {env_id}")
    print(f"Checkpoint: {target_model.name}")

    try:
        succ_rate, avg_ret = evaluate_checkpoint(str(target_model), env_id)
        results.append({
            "Task": env_id,
            "Decay Schedule": "Online",
            "Checkpoint": target_model.name,
            "Success Rate (%)": f"{succ_rate:.1f}%",
            "Avg Reward": f"{avg_ret:.2f}"
        })
        
        df = pd.DataFrame(results)
        df.to_csv("thesis_results.csv", index=False)
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()