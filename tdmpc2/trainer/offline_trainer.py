import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob
from tdmpc2.utils import get_distillation_coefficient

# import cProfile
# import pstats
# from pstats import SortKey
from line_profiler import LineProfiler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA

import wandb
from common.buffer import Buffer
from trainer.base import Trainer

def profile_with_line_profiler(func):
    def wrapper(*args, **kwargs):
        prof = LineProfiler()
        try:
            return prof(func)(*args, **kwargs)
        finally:
            prof.print_stats()
    return wrapper


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
		self.device = next(self.agent.model.parameters()).device
	
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		results = dict()
		for task_idx in tqdm(range(len(self.cfg.tasks)), desc='Evaluating'):
			ep_rewards, ep_successes = [], []
			for _ in range(self.cfg.eval_episodes):
				obs, done, ep_reward, t = self.env.reset(task_idx), False, 0, 0
				while not done:
					action = self.agent.act(obs, t0=t==0, eval_mode=True, task=task_idx)
					obs, reward, done, info = self.env.step(action)
					ep_reward += reward
					t += 1
				ep_rewards.append(ep_reward)
				ep_successes.append(info['success'])
			results.update({
				f'episode_reward+{self.cfg.tasks[task_idx]}': np.nanmean(ep_rewards),
				f'episode_success+{self.cfg.tasks[task_idx]}': np.nanmean(ep_successes),})
		return results
				
	def train(self):
		"""Train a TD-MPC2 agent."""
		assert self.cfg.multitask and self.cfg.task in {'mt30', 'mt80'}, \
			'Offline training only supports multitask training with mt30 or mt80 task sets.'

		# Load data
		assert self.cfg.task in self.cfg.data_dir, \
			f'Expected data directory {self.cfg.data_dir} to contain {self.cfg.task}, ' \
			f'please double-check your config.'
		fp = Path(os.path.join(self.cfg.data_dir, '*.pt'))
		fps = sorted(glob(str(fp)))
		assert len(fps) > 0, f'No data found at {fp}'
		print(f'Found {len(fps)} files in {fp}')
	
		# Create buffer for sampling
		_cfg = deepcopy(self.cfg)
		_cfg.episode_length = 101 if self.cfg.task == 'mt80' else 501
		_cfg.buffer_size = 380_450_000 if self.cfg.task == 'mt80' else 345_690_000 # 550_450_000 mt80
		_cfg.steps = _cfg.buffer_size  # Match the buffer size for full GPU utilization
		self.buffer = Buffer(_cfg)

		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp)
			assert td.shape[1] == _cfg.episode_length, \
				f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
				f'please double-check your config.'
			for i in range(len(td)):
				self.buffer.add(td[i])
		# assert self.buffer.num_eps == self.buffer.capacity, \
		# 	f'Buffer has {self.buffer.num_eps} episodes, expected {self.buffer.capacity} episodes.'
		
		print(f'Loaded {self.buffer.num_eps} episodes into buffer with capacity {self.buffer.capacity}')
		
		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
  
		# self.distill_initialization()
  
		for i in tqdm(range(self.cfg.steps)):
			train_metrics = self.agent.update(self.buffer, step=i) #self.update_agent()
			if i != 0 and i % 50000 == 0 or i % 337000 == 0:
				self.logger.save_agent(self.agent, identifier=f'{i}')
    
		self.logger.finish(self.agent)