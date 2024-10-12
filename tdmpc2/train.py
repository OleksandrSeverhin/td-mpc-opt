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
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

torch.backends.cudnn.benchmark = True

# config_multi_distill, config_single_distill, config
@hydra.main(config_name='generic', config_path='.') # config
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
	# assert cfg.steps > 0, 'Must train for at least 1 step.'
 
	### optional teacher for distill
	# cfg_teacher = parse_cfg(cfg.model2)
	# set_seed(cfg_teacher.seed)
	# env = make_env(cfg_teacher)
	# buffer = Buffer(cfg_teacher)
	# teacher_model = TDMPC2(cfg_teacher)
	# teacher_model.load(cfg_teacher.checkpoint)  # Load 317M pretrained 

 
	cfg_student = parse_cfg(cfg.model1)
	set_seed(cfg_student.seed)
	env = make_env(cfg_student)
	buffer = Buffer(cfg_student)
	# student_model = TDMPC2(cfg_student, teacher_model) # distill
	student_model = TDMPC2(cfg_student) #, teacher_model) # from scratch
	# state_dict = torch.load('/home/dmytrok/rl_exp/tdmpc2-opt/tdmpc2/logs/mt30/1/mt30_317to1M_200k_steps_noQ_buffer60k_dist0_4_20240923/models/final.pt')
	# student_model.model.load_state_dict(state_dict['model'])
 
	### Pretrain
	# student_model.load('/home/dmytrok/rl_exp/tdmpc2-opt/tdmpc2/logs/mt30/1/mt30_fromscratch_1M_BS1024_337K_steps/models/200000.pt')
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg_student.work_dir)

	trainer = OfflineTrainer(
		cfg=cfg_student,
		env=env,
		agent=student_model,
		buffer=buffer,
		logger=Logger(cfg_student)
	)

	# trainer.collect_teacher_latents()
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
