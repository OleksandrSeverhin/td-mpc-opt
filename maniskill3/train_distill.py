import os
import copy
import torch
import multiprocessing
import hydra
import gymnasium as gym
from termcolor import colored
from omegaconf import OmegaConf

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_envs
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger, print_run

torch.backends.cudnn.benchmark = True

@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
    assert torch.cuda.is_available()
    
    # 1. Setup Student Configuration
    set_seed(cfg.seed)
    student_cfg = parse_cfg(cfg) 

    # 2. Initialize ManiSkill3 Env to get Student shapes
    print(colored('Initializing ManiSkill3 Environment...', 'yellow'))
    temp_env = make_envs(student_cfg, student_cfg.num_envs)
    obs_space = temp_env.observation_space
    if isinstance(obs_space, gym.spaces.Dict):
        student_cfg.obs_shape = {k: v.shape for k, v in obs_space.spaces.items()}
    else:
        student_cfg.obs_shape = {student_cfg.obs: obs_space.shape}
    student_cfg.action_dim = temp_env.action_space.shape[0]
    student_cfg.episode_length = temp_env.max_episode_steps
    student_cfg.action_dims = [student_cfg.action_dim]
    student_cfg.episode_lengths = [student_cfg.episode_length]
    
    # 3. Setup Teacher Configuration (Hardcoded to MT30-317M Specs)
    print(colored('Preparing Teacher Configuration (MT30-317M)...', 'blue'))
    teacher_cfg = copy.deepcopy(student_cfg)
    
    # These parameters MUST match the mt30-317M.pt checkpoint exactly
    teacher_cfg.model_size = 317
    teacher_cfg.num_enc_layers = 5
    teacher_cfg.enc_dim = 4096
    teacher_cfg.mlp_dim = 4096
    teacher_cfg.latent_dim = 1376
    teacher_cfg.task_dim = 96      
    teacher_cfg.num_q = 8          
    teacher_cfg.multitask = True
    
    # Override shapes to MT30 native dimensions for weight loading
    teacher_cfg.obs_shape = {'state': (120,)} # MT30 state dim is 120
    teacher_cfg.action_dim = 12               # MT30 action dim is 12
    teacher_cfg.action_dims = [12] * 30       # MT30 has 30 tasks
    teacher_cfg.episode_lengths = [500] * 30
    teacher_cfg.true_latent_dim = 1376 

    # 4. Initialize Models
    print(colored('Loading Teacher Weights...', 'blue', attrs=['bold']))
    teacher_model = TDMPC2(teacher_cfg)
    # Load weights with strict=True now that architecture matches
    teacher_model.load(student_cfg.checkpoint, strict=True)
    
    teacher_model.model.eval()
    for p in teacher_model.model.parameters():
        p.requires_grad = False
    
    print(colored(f'Initializing Student ({student_cfg.model_size}M)...', 'green'))
    agent = TDMPC2(student_cfg, teacher=teacher_model)
    
    # 5. Infrastructure Setup
    manager = multiprocessing.Manager()
    logger = Logger(student_cfg, manager)
    video_path = student_cfg.work_dir / 'eval_video'
    os.makedirs(video_path, exist_ok=True)
    
    eval_env = make_envs(student_cfg, student_cfg.num_eval_envs, 
                         video_path=video_path, is_eval=True, logger=logger)
    
    print_run(student_cfg)

    # 6. Run Trainer
    trainer = OnlineTrainer(
        cfg=student_cfg,
        env=temp_env,
        eval_env=eval_env,
        agent=agent,
        buffer=Buffer(student_cfg),
        logger=logger,
    )
    trainer.train()

if __name__ == '__main__':
    train()