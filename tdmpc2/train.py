import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'

import warnings
warnings.filterwarnings('ignore')

import torch
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

import hydra
from omegaconf import DictConfig
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from common.logger import Logger


@hydra.main(config_name='generic', config_path='.')
def train(cfg: DictConfig):
    """
    Train a single-task or multi-task TD-MPC2 agent.

    Example usage:
        $ python train.py task=mt30 model_size=317
        $ python train.py task=dog-run steps=7_000_000
    """

    assert torch.cuda.is_available(), "CUDA is not available. Please check your GPU setup."

    # Load and initialize teacher
    cfg_teacher = parse_cfg(cfg.teacher_config)
    set_seed(cfg_teacher.seed)
    env = make_env(cfg_teacher)
    buffer = Buffer(cfg_teacher)
    teacher_model = TDMPC2(cfg_teacher)
    teacher_model.load(cfg_teacher.checkpoint)

    # Load and initialize student
    cfg_student = parse_cfg(cfg.student_config)
    set_seed(cfg_student.seed)
    env = make_env(cfg_student)
    buffer = Buffer(cfg_student)
    student_model = TDMPC2(cfg_student, teacher_model=teacher_model)

    print(colored('Work dir:', 'yellow', attrs=['bold']), cfg_student.work_dir)

    # Offline training
    trainer = OfflineTrainer(
        cfg=cfg_student,
        env=env,
        agent=student_model,
        buffer=buffer,
        logger=Logger(cfg_student)
    )

    trainer.train()
    print('\nTraining completed successfully')


if __name__ == '__main__':
    train()
