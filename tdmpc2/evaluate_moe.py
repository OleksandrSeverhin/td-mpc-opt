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
import torch
from termcolor import colored
from tqdm import tqdm

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from omegaconf import OmegaConf
from tdmpc2 import TDMPC2

torch.backends.cudnn.benchmark = True

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

@hydra.main(config_name='generic', config_path='.')
def evaluate(cfg: dict):
    """
    Custom Evaluation Script specifically for the MT30 MoE Student.
    """
    assert torch.cuda.is_available()
    
    # 1. Forcefully inject MoE architecture parameters (Bypassing Hydra)
    OmegaConf.set_struct(cfg, False)
    
    # Extract the student configuration if running through generic.yaml
    if hasattr(cfg, 'student_config'):
        eval_cfg = parse_cfg(cfg.student_config)
        OmegaConf.set_struct(eval_cfg, False)
    else:
        eval_cfg = parse_cfg(cfg)
        
    eval_cfg.is_moe_student = True
    eval_cfg.latent_dim = 1376
    eval_cfg.mlp_dim = 4096
    
    # Ensure evaluation settings
    if not hasattr(eval_cfg, 'eval_episodes'):
        eval_cfg.eval_episodes = 10
    if not hasattr(eval_cfg, 'save_video'):
        eval_cfg.save_video = True

    set_seed(eval_cfg.seed)

    print(colored('====================================', 'green'))
    print(colored('   INITIALIZING MOE EVALUATION      ', 'green'))
    print(colored('====================================', 'green'))
    print(colored(f'Task: {eval_cfg.task}', 'blue', attrs=['bold']))
    print(colored(f'Architecture: MoE Student (Latent: {eval_cfg.latent_dim})', 'blue', attrs=['bold']))
    print(colored(f'Checkpoint: {eval_cfg.checkpoint}', 'blue', attrs=['bold']))

    # 2. Make environment
    env = make_env(eval_cfg)

    # 3. Load agent (Will safely build MoEPolicy due to injected flags)
    agent = TDMPC2(eval_cfg)

    assert os.path.exists(eval_cfg.checkpoint), f'Checkpoint {eval_cfg.checkpoint} not found! Must provide a valid absolute path.'
    agent.load(eval_cfg.checkpoint)
    agent.model.eval()
    
    # 4. Evaluate Loop
    if eval_cfg.multitask:
        print(colored(f'Evaluating MoE agent on {len(eval_cfg.tasks)} tasks:', 'yellow', attrs=['bold']))
    else:
        print(colored(f'Evaluating MoE agent on {eval_cfg.task}:', 'yellow', attrs=['bold']))
        
    if eval_cfg.save_video:
        video_dir = os.path.join(eval_cfg.work_dir, 'videos_eval')
        os.makedirs(video_dir, exist_ok=True)
        
    scores = []
    tasks = eval_cfg.tasks if eval_cfg.multitask else [eval_cfg.task]

    for task_idx, task in enumerate(tasks):
        start_time = time.time()
        if not eval_cfg.multitask:
            task_idx = None
        ep_rewards, ep_successes = [], []
        
        for i in range(eval_cfg.eval_episodes):
            obs, done, ep_reward, t = env.reset(task_idx=task_idx), False, 0, 0
            if eval_cfg.save_video:
                frames = [env.render()]

            while not done:
                # Ask the student model for an action in evaluation mode
                with torch.no_grad():
                    action = agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                t += 1
                if eval_cfg.save_video:
                    frames.append(env.render())
                    
            ep_rewards.append(ep_reward)
            ep_successes.append(info.get('success', 0.0))

            if eval_cfg.save_video:
                vid_path = os.path.join(video_dir, f'{task}_ep{i}.mp4')
                imageio.mimsave(vid_path, frames, fps=15)

        ep_rewards = np.mean(ep_rewards)
        ep_successes = np.mean(ep_successes)

        if eval_cfg.multitask:
            scores.append(ep_successes * 100 if task.startswith('mw-') else ep_rewards / 10)
            
        print(colored(f'  {task:<22}' \
            f'\tR: {ep_rewards:.01f}  ' \
            f'\tS: {ep_successes:.02f}', 'yellow'))

    if eval_cfg.multitask:
        print(colored(f'Normalized MT30 MoE score: {np.mean(scores):.02f}', 'green', attrs=['bold']))

if __name__ == '__main__':
    evaluate()