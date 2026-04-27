import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_envs
from tdmpc2 import TDMPC2

@hydra.main(config_name='config', config_path='.', version_base=None)
def evaluate(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    
    OmegaConf.set_struct(cfg, False)
    base_cfg = parse_cfg(cfg) 

    print(colored('Initializing ManiSkill3 Evaluation Environment...', 'yellow'))
    env = make_envs(base_cfg, 1, is_eval=True)
    
    print(colored("Configuring Teacher's MT30 Architecture...", 'green'))
    teacher_cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    teacher_cfg_dict['model_size'] = 317
    teacher_cfg_dict['num_enc_layers'] = 5
    teacher_cfg_dict['enc_dim'] = 4096
    teacher_cfg_dict['mlp_dim'] = 4096
    teacher_cfg_dict['num_q'] = 8
    teacher_cfg_dict['latent_dim'] = 1376 
    teacher_cfg_dict['true_latent_dim'] = 1376
    teacher_cfg_dict['obs_shape'] = {'state': [24]} 
    teacher_cfg_dict['action_dim'] = 6 
    teacher_cfg_dict['action_dims'] = [6] * 30
    teacher_cfg_dict['multitask'] = True
    teacher_cfg_dict['task_dim'] = 96
    teacher_cfg_dict['tasks'] = [str(i) for i in range(30)]
    teacher_cfg = OmegaConf.create(teacher_cfg_dict)

    print(colored('Loading Teacher Weights...', 'blue'))
    teacher_model = TDMPC2(teacher_cfg)
    teacher_model.load(base_cfg.checkpoint, strict=True)
    teacher_model.model.eval()
    
    print(colored(f'Running Baseline on {base_cfg.env_id}...', 'cyan'))
    ep_rewards = []
    ep_successes = []

    episodes = 12500
    for ep in range(episodes):
        # Handle Gym/ManiSkill3 return formats
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        
        done = False
        ep_reward = 0
        success = 0

        while not done:
            # 1. Slice observation to 24-dim for MT30 Teacher
            t_obs = obs[:, :24]
            # 2. Provide "Task 0" dummy task
            t_task = torch.zeros(t_obs.shape[0], dtype=torch.long, device=teacher_model.device)

            with torch.no_grad():
                t_action = teacher_model.act(t_obs, eval_mode=True, task=t_task)

            # 3. Pad Teacher's 6-dim action back to ManiSkill3's 8-dim action space
            action = torch.zeros((1, env.action_space.shape[-1]), dtype=torch.float32)
            action[:, :6] = t_action

            # Step environment
            step_out = env.step(action)
            # Handle variable step return lengths
            if len(step_out) == 4:
                obs, reward, done, info = step_out
            else:
                obs, reward, terminated, truncated, info = step_out
                done = terminated.any() or truncated.any() if isinstance(terminated, torch.Tensor) else terminated or truncated
            
            # Aggregate rewards
            # Depending on wrapper, reward might be a tensor or scalar
            r = reward[0].item() if hasattr(reward, 'item') else reward
            ep_reward += r

            # Check for ManiSkill3 success metric
            if isinstance(info, dict) and 'success' in info:
                is_success = info['success'][0].item() if hasattr(info['success'], 'item') else info['success']
                if is_success:
                    success = 1

        ep_rewards.append(ep_reward)
        ep_successes.append(success)
        print(f"Episode {ep+1} | Reward: {ep_reward:.2f} | Success: {success}")

    print(colored('========================================', 'magenta'))
    print(colored(f'TEACHER BASELINE ({base_cfg.env_id})', 'magenta'))
    print(colored(f'Average Reward:  {np.mean(ep_rewards):.2f}', 'magenta'))
    print(colored(f'Success Rate:    {np.mean(ep_successes):.2f}', 'magenta'))
    print(colored('========================================', 'magenta'))

if __name__ == '__main__':
    evaluate()