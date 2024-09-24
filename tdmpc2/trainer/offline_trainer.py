import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

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
		_cfg.buffer_size = 345_690_000  # 60M steps
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
			train_metrics = self.agent.update(self.buffer) #self.update_agent()
			if i != 0 and i % 5000 == 0:
				self.logger.save_agent(self.agent, identifier=f'{i}')
    
		self.logger.finish(self.agent)


class DistillationOfflineTrainer(OfflineTrainer):
	def __init__(self, cfg, env, agent, buffer, logger, teacher_model):
		super().__init__(cfg, env, agent, buffer, logger)
		self.teacher_model = teacher_model
		self.distillation_weight = cfg.distillation_weight
		self.distillation_temperature = cfg.distillation_temperature
		
		self.device = next(self.agent.model.parameters()).device

	# @profile_with_line_profiler
	def update_agent(self):
		batch = self.buffer.sample()
		obs, action, reward, task = batch 

		with torch.no_grad():
			teacher_z = self.teacher_model.model.encode(obs[0].to(self.device), task)
			teacher_reward = self.teacher_model.model.reward(teacher_z, action[0].to(self.device), task)
			teacher_q = self.teacher_model.model.Q(teacher_z, action[0].to(self.device), task)

		student_z = self.agent.model.encode(obs[0].to(self.device), task)
		student_reward = self.agent.model.reward(student_z, action[0].to(self.device), task)
		student_q = self.agent.model.Q(student_z, action[0].to(self.device), task)

		reward_loss = F.mse_loss(student_reward, teacher_reward)
		q_loss = F.kl_div(
			F.log_softmax(student_q / self.distillation_temperature, dim=-1),
			F.softmax(teacher_q / self.distillation_temperature, dim=-1),
			reduction='batchmean'
		) * (self.distillation_temperature ** 2)
  
		dist_loss = reward_loss + q_loss
		
		original_loss_dict = self.agent.update(obs, action, reward, task)

		alpha = self.distillation_weight
		total_loss = original_loss_dict['total_loss'] + alpha * dist_loss

		self.agent.optim.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.agent.cfg.grad_clip_norm)
		self.agent.optim.step()

		original_loss_dict.update({
			'distillation_loss': dist_loss.item(),
			'total_loss': total_loss.item()
		})

		wandb.log({
			'distillation_loss': dist_loss.item(),
			'total_loss': total_loss.item(),
		})

		return original_loss_dict