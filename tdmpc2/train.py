import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch
torch.autograd.set_detect_anomaly(True)

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
# from tdmpc2_orig import TDMPC2
from trainer.offline_trainer import OfflineTrainer, DistillationOfflineTrainer
from trainer.online_trainer import OnlineTrainer, DistillationOnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True

# config_multi_distill, config_single_distill, config
@hydra.main(config_name='config_multi_distill', config_path='.') # config
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	env = make_env(cfg)
	buffer = Buffer(cfg)
	cfg.steps = 200000  # Set total steps to 200k
	cfg.buffer_size = 20000  # Set buffer size to 20k

	teacher_model = TDMPC2(cfg)
	teacher_model.load(cfg.checkpoint)  # Load 317M pretrained checkpoint

	cfg.model_size = 1  # Set student model size to 1M
	cfg.num_enc_layers = 2
	cfg.enc_dim = 256
	cfg.mlp_dim = 384
	cfg.latent_dim = 128
	cfg.num_q = 2

	student_model = TDMPC2(cfg)  # 1M non-trained model

	trainer = DistillationOfflineTrainer(
		cfg=cfg,
		env=env,
		agent=student_model,
		buffer=buffer,
		logger=Logger(cfg),
		teacher_model=teacher_model
	)

	# trainer.collect_teacher_latents()
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
