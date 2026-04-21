import os
import torch
import multiprocessing
import hydra
from omegaconf import OmegaConf
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_envs
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True

@hydra.main(config_name='config', config_path='.', version_base=None)
def train(cfg: dict):
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    
    OmegaConf.set_struct(cfg, False)
    base_cfg = parse_cfg(cfg) 

    print(colored('Initializing ManiSkill3 Environment...', 'yellow'))
    temp_env = make_envs(base_cfg, base_cfg.num_envs)
    
    print(colored('Configuring Student (1M) for ManiSkill3...', 'green'))
    student_cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    student_cfg_dict['obs_shape'] = {'state': [temp_env.observation_space.shape[-1]]} 
    student_cfg_dict['action_dim'] = temp_env.action_space.shape[-1]
    student_cfg_dict['action_dims'] = [student_cfg_dict['action_dim']]
    student_cfg = OmegaConf.create(student_cfg_dict)

    teacher_cfg_dict = OmegaConf.to_container(student_cfg, resolve=True)
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
    for p in teacher_model.model.parameters():
        p.requires_grad = False
    
    print(colored(f'Initializing Student (1M) with {temp_env.observation_space.shape[-1]} inputs...', 'green'))
    agent = TDMPC2(student_cfg, teacher=teacher_model)
    
    trainer = OnlineTrainer(
        cfg=student_cfg,
        env=temp_env,
        eval_env=make_envs(student_cfg, student_cfg.num_eval_envs, is_eval=True),
        agent=agent,
        buffer=Buffer(student_cfg),
        logger=Logger(student_cfg, multiprocessing.Manager()),
    )
    trainer.train()

if __name__ == '__main__':
    train()