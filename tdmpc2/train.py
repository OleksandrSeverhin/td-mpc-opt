import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import torch.nn.functional as F
from tensordict.tensordict import TensorDict
from termcolor import colored
import hydra
from omegaconf import DictConfig, OmegaConf
from huggingface_hub import HfApi

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from common.logger import Logger
from trainer.distill_trainer import DistillTrainer 

TASKS = [
    'walker-stand', 'walker-walk', 'walker-run', 'cheetah-run', 'reacher-easy', 
    'reacher-hard', 'acrobot-swingup', 'pendulum-swingup', 'cartpole-balance', 
    'cartpole-balance-sparse', 'cartpole-swingup', 'cartpole-swingup-sparse', 
    'cup-catch', 'finger-spin', 'finger-turn-easy', 'finger-turn-hard', 
    'fish-swim', 'hopper-stand', 'hopper-hop', 'walker-walk-backwards', 
    'walker-run-backwards', 'cheetah-run-backwards', 'cheetah-run-front', 
    'cheetah-run-back', 'cheetah-jump', 'hopper-hop-backwards', 
    'reacher-three-easy', 'reacher-three-hard', 'cup-spin', 'pendulum-spin'
]

def to_td(obs, action=None, reward=None, done=None, task=None):
    """
    Standardizes dimensions for TD-MPC2 episodic storage.
    """
    if isinstance(task, str):
        task_idx = TASKS.index(task)
    else:
        task_idx = task if task is not None else 0

    # Ensure obs and action are at least 2D (1, Dim)
    obs = obs.unsqueeze(0).cpu() if obs.dim() == 1 else obs.cpu()
    
    if action is not None:
        action = action.unsqueeze(0).cpu() if action.dim() == 1 else action.cpu()
    else:
        action = torch.full((1, 6), float('nan'))

    # Scalars (reward, done, task) should be shape (1,) for batch_size=(1,)
    reward = torch.tensor([reward], dtype=torch.float32) if reward is not None else torch.tensor([float('nan')])
    done = torch.tensor([done], dtype=torch.bool) if done is not None else torch.tensor([False])
    task_tensor = torch.tensor([task_idx], dtype=torch.long)

    return TensorDict({
        'obs': obs,
        'action': action,
        'reward': reward,
        'done': done,
        'task': task_tensor
    }, batch_size=(1,))

@hydra.main(config_name='generic', config_path='.')
def train(cfg: DictConfig):
    assert torch.cuda.is_available(), "CUDA is not available."
    OmegaConf.set_struct(cfg, False)

    # 1. Load Teacher
    cfg_teacher = parse_cfg(cfg.teacher_config)
    set_seed(cfg_teacher.seed)
    teacher_model = TDMPC2(cfg_teacher)
    teacher_model.load(cfg_teacher.checkpoint)
    teacher_model.model.eval() 

    # 2. Load MoE Student
    cfg_student = parse_cfg(cfg.student_config)
    cfg_student.is_moe_student = True 
    set_seed(cfg_student.seed)
    
    env_student = make_env(cfg_student)
    buffer = Buffer(cfg_student)
    student_model = TDMPC2(cfg_student, teacher_model=teacher_model)
    
    logger = Logger(cfg_student)
    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg_student.work_dir)

    # 3. Initialize DistillTrainer
    teachers_dict = {i: teacher_model for i in range(30)} 
    trainer = DistillTrainer(
        student=student_model,
        teachers=teachers_dict, 
        alpha=cfg_student.latent_distill_alpha
    )

    # 4. Data Collection & Training Loop
    obs, done, t = env_student.reset(), False, 0
    raw_task = getattr(env_student, 'task', 0)
    curr_task_idx = TASKS.index(raw_task) if isinstance(raw_task, str) else raw_task
    
    _tds = [to_td(obs, task=curr_task_idx)] 
    seed_steps = cfg_student.batch_size * 2 

    print(colored('Starting training loop...', 'green'))
    for step in range(cfg_student.steps):
        if step < seed_steps:
            action = torch.tensor(env_student.action_space.sample(), dtype=torch.float32)
        else:
            with torch.no_grad():
                task_tensor = torch.tensor([curr_task_idx], device='cuda', dtype=torch.long)
                action = student_model.act(obs, t0=(t==0), eval_mode=False, task=task_tensor) 

        next_obs, reward, done, info = env_student.step(action)
        
        if isinstance(done, (list, tuple, np.ndarray)):
            done = any(done)

        _tds.append(to_td(next_obs, action, reward, done, curr_task_idx))
        
        obs = next_obs
        t += 1
        
        if done:
            ep_td = torch.cat(_tds)
            buffer.add(ep_td)
            
            obs, done, t = env_student.reset(), False, 0
            raw_task = getattr(env_student, 'task', 0)
            curr_task_idx = TASKS.index(raw_task) if isinstance(raw_task, str) else raw_task
            _tds = [to_td(obs, task=curr_task_idx)] 
            
        if step >= seed_steps and buffer._num_eps > 0:
            try:
                metrics = trainer.update(buffer)
                if step % 1000 == 0:
                    print(f"Step {step} | Total Loss: {metrics['total_loss']:.4f}")
                    logger.log(metrics)
                    
                if step % 500000 == 0 or step == cfg_student.steps - 1:
                    save_dir = os.path.join(cfg_student.work_dir, 'models')
                    os.makedirs(save_dir, exist_ok=True)
                    local_path = os.path.join(save_dir, f'step_{step}.pt')
                    
                    student_model.save(local_path)
                    print(colored(f"Model saved locally at step {step}", 'green'))
                        
            except Exception as e:
                print(colored(f"Training failed at step {step}: {e}", 'red'))
                break

    print(colored('Process complete.', 'cyan'))

if __name__ == '__main__':
    train()
