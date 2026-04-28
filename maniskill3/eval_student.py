import torch
import numpy as np
import hydra
from omegaconf import OmegaConf

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_envs
from tdmpc2 import TDMPC2

@hydra.main(config_name='config', config_path='.', version_base=None)
def evaluate_student(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    
    OmegaConf.set_struct(cfg, False)
    base_cfg = parse_cfg(cfg) 

    # Enforce Student Model parameters (1M)
    base_cfg.model_size = 1
    base_cfg.num_enc_layers = 2
    base_cfg.enc_dim = 256
    base_cfg.mlp_dim = 512
    base_cfg.latent_dim = 512
    
    env = make_envs(base_cfg, 1, is_eval=True)

    model = TDMPC2(base_cfg)
    model.load(base_cfg.checkpoint)
    model.model.eval()
    
    ep_rewards = []
    ep_successes = []

    # Run 100 Evaluation Episodes
    for ep in range(12500):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
        done = False
        ep_reward = 0
        success = 0

        while not done:
            with torch.no_grad():
                # Student evaluates directly, no task fallback needed for purely single-task
                action = model.act(obs, eval_mode=True)

            step_out = env.step(action)
            if len(step_out) == 4:
                obs, reward, done, info = step_out
            else:
                obs, reward, terminated, truncated, info = step_out
                done = terminated.any() or truncated.any() if isinstance(terminated, torch.Tensor) else terminated or truncated
            
            r = reward[0].item() if hasattr(reward, 'item') else reward
            ep_reward += r

            if isinstance(info, dict) and 'success' in info:
                is_success = info['success'][0].item() if hasattr(info['success'], 'item') else info['success']
                if is_success:
                    success = 1

        ep_rewards.append(ep_reward)
        ep_successes.append(success)
        print(f"Episode {ep+1}/100 | Reward: {ep_reward:.2f} | Success: {success}")

    print('========================================')
    print(f'STUDENT BASELINE ({base_cfg.env_id})')
    print(f'Average Reward:  {np.mean(ep_rewards):.2f}')
    print(f'Success Rate:    {np.mean(ep_successes):.2f}')
    print('========================================')

if __name__ == '__main__':
    evaluate_student()