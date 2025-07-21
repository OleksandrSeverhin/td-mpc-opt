import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'

import warnings
warnings.filterwarnings('ignore')

import torch
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

@hydra.main(config_name='generic', config_path='.')  # config
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Usage examples:
		$ python train.py task=mt30 model_size=317
	"""

	assert torch.cuda.is_available(), "CUDA is not available"
	device = torch.device("cuda")

	# ---- Optional Teacher Model ----
	cfg_teacher = parse_cfg(cfg.teacher_config)
	set_seed(cfg_teacher.seed)

	env = make_env(cfg_teacher)
	buffer = Buffer(cfg_teacher)
	teacher_model = TDMPC2(cfg_teacher)
	teacher_model.load(cfg_teacher.checkpoint)
	teacher_model.to(device)  # Move teacher to GPU

	# ---- Student Model ----
	cfg_student = parse_cfg(cfg.student_config)
	set_seed(cfg_student.seed)

	env = make_env(cfg_student)
	buffer = Buffer(cfg_student)
	student_model = TDMPC2(cfg_student, teacher_model)
	student_model.to(device)  # Move student to GPU

	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg_student.work_dir)

	trainer = OfflineTrainer(
		cfg=cfg_student,
		env=env,
		agent=student_model,
		buffer=buffer,
		logger=Logger(cfg_student)
	)

	# Uncomment if using teacher's latents
	# trainer.collect_teacher_latents()
	trainer.train()

	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
