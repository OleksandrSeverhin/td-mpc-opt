import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
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

	teacher_model = TDMPC2(cfg) # to change
	teacher_model.load(cfg.checkpoint) # 5M pretrained reacher-three-hard

	cfg.model_size = 1 # only for multi
	cfg.mlp_dim = 384 # 384 # 512 
	cfg.latent_dim = 128 # 128 # 512
	cfg.num_q = 2 # 2 # 5
	# distillation
	cfg.distillation_temperature = 2.0
	cfg.distillation_weight = 0.5

	student_model = TDMPC2(cfg) # 1M non-trained

	# change the trainer beforehand
	trainer = DistillationOfflineTrainer(
        cfg=cfg,
        env=env,
        agent=student_model, # 1M or something, new smaller arch
        buffer=buffer,
        logger=Logger(cfg),
        teacher_model=teacher_model # 5M pretrain
    )

	# trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	# trainer = trainer_cls(
	# 	cfg=cfg,
	# 	env=make_env(cfg),
	# 	agent=TDMPC2(cfg),
	# 	buffer=Buffer(cfg),
	# 	logger=Logger(cfg),
	# )
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
