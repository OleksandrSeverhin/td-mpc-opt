import os
from copy import deepcopy
from time import time
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from common.buffer import Buffer
from trainer.base import Trainer


class OfflineTrainer(Trainer):
	"""Trainer class for multi-task offline TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._start_time = time()
	
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
		## Changed
		_cfg.buffer_size = min(30_000_000, 345_690_000)  # Use the smaller of the two values
		_cfg.steps = _cfg.buffer_size  # Match the buffer size for full GPU utilization
		self.buffer = Buffer(_cfg)
		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp)
			assert td.shape[1] == _cfg.episode_length, \
				f'Expected episode length {td.shape[1]} to match config episode length {_cfg.episode_length}, ' \
				f'please double-check your config.'
			for i in range(len(td)):
				if self.buffer.num_eps >= self.buffer.capacity:
					break
				self.buffer.add(td[i])
			if self.buffer.num_eps >= self.buffer.capacity:
				break
		
		print(f'Loaded {self.buffer.num_eps} episodes into buffer with capacity {self.buffer.capacity}')
		
		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		for i in tqdm(range(self.cfg.steps)):
			# Update agent using distillation
			train_metrics = self.update_agent()
			
			if i != 0 and i % 100000 == 0:
				self.logger.save_agent(self.agent, identifier=f'{i}')
		self.logger.finish(self.agent)
			# Update agent
			# train_metrics = self.agent.update(self.buffer)

		# 	# Evaluate agent periodically
		# 	if i == self.cfg.eval_freq: # or i % 10_000 == 0:
		# 		metrics = {
		# 			'iteration': i,
		# 			'total_time': time() - self._start_time,
		# 		}
		# 		metrics.update(train_metrics)
		# 		if i == self.cfg.eval_freq: # i % self.cfg.eval_freq == 0:
		# 			metrics.update(self.eval())
		# 			self.logger.pprint_multitask(metrics, self.cfg)
		# 			if i > 0:
		# 				self.logger.save_agent(self.agent, identifier=f'{i}')
		# 		self.logger.log(metrics, 'pretrain')
			
		# self.logger.finish(self.agent)

def distillation_loss(student_outputs, teacher_outputs, temperature=2.0):
    """
    Compute the knowledge distillation loss using KL divergence.
    """
    return F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=-1),
        F.softmax(teacher_outputs / temperature, dim=-1),
        reduction='batchmean',
        log_target=False
    ) * (temperature ** 2)


class DistillationOfflineTrainer(OfflineTrainer):
    def __init__(self, cfg, env, agent, buffer, logger, teacher_model):
        super().__init__(cfg, env, agent, buffer, logger)
        self.teacher_model = teacher_model
        self.distillation_weight = cfg.distillation_weight
        self.distillation_temperature = cfg.distillation_temperature
        
        self.device = next(self.agent.model.parameters()).device
        self.teacher_projection = nn.Linear(512, 128).to(self.device)  # 5M > 1M latent space, 512 > 128
        
        # Initialize the projection layer
        nn.init.xavier_uniform_(self.teacher_projection.weight)
        nn.init.zeros_(self.teacher_projection.bias)

    def update_agent(self):
        batch = self.buffer.sample()
        obs, action, reward, task = batch 

        # Get teacher outputs
        with torch.no_grad():
            teacher_z = self.teacher_model.model.encode(obs[0].to(self.device), task)
            teacher_next_z = self.teacher_model.model.next(teacher_z, action[0].to(self.device), task)
            teacher_next_z_projected = self.teacher_projection(teacher_next_z)
            teacher_reward = self.teacher_model.model.reward(teacher_z, action[0].to(self.device), task)

        # Get student outputs
        student_z = self.agent.model.encode(obs[0].to(self.device), task)
        student_next_z = self.agent.model.next(student_z, action[0].to(self.device), task)
        student_reward = self.agent.model.reward(student_z, action[0].to(self.device), task)

        # Compute distillation loss
        next_z_dist_loss = distillation_loss(student_next_z, teacher_next_z_projected, self.distillation_temperature)
        reward_dist_loss = distillation_loss(student_reward, teacher_reward, self.distillation_temperature)
        dist_loss = next_z_dist_loss + reward_dist_loss

        # Compute original TD-MPC2 loss
        original_loss_dict = self.agent.update(obs, action, reward, task)
        
        # Combine losses
        total_loss = original_loss_dict['total_loss'] + self.distillation_weight * dist_loss

        # Backward pass
        self.agent.optim.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.agent.cfg.grad_clip_norm)
        
        # Optimize
        self.agent.optim.step()

        # Update the original loss dictionary with the distillation loss
        original_loss_dict['distillation_loss'] = dist_loss.item()
        original_loss_dict['total_loss'] = total_loss.item()

        return original_loss_dict