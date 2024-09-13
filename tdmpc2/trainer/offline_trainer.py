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
from sklearn.decomposition import PCA

import wandb
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
		_cfg.buffer_size = 5_000_000  # 60M steps
		_cfg.steps = _cfg.buffer_size  # Match the buffer size for full GPU utilization
		self.buffer = Buffer(_cfg)

		# Calculate the number of episodes to sample from each file
		total_episodes = sum(torch.load(fp).shape[0] for fp in fps)
		episodes_per_file = {fp: torch.load(fp).shape[0] for fp in fps}
		sample_ratio = _cfg.buffer_size / (total_episodes * _cfg.episode_length)

		# Sample episodes uniformly from each file
		for fp in tqdm(fps, desc='Loading data'):
			td = torch.load(fp)
			num_episodes_to_sample = int(episodes_per_file[fp] * sample_ratio)
			sampled_indices = np.random.choice(td.shape[0], num_episodes_to_sample, replace=False)
			for idx in sampled_indices:
				if self.buffer.num_eps >= self.buffer.capacity:
					break
				self.buffer.add(td[idx])
			if self.buffer.num_eps >= self.buffer.capacity:
				break
		
		print(f'Loaded {self.buffer.num_eps} episodes into buffer with capacity {self.buffer.capacity}')
		
		print(f'Training agent for {self.cfg.steps} iterations...')
		metrics = {}
		for i in tqdm(range(self.cfg.steps)):
			# Update agent using distillation
			if not self.pca_fitted and self.buffer_ready_for_pca():
				self.fit_pca()
    
			train_metrics = self.update_agent()
			
			if i != 0 and i % 10000 == 0:
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

# def distillation_loss(student_outputs, teacher_outputs, temperature=2.0):
#     """
#     Compute the knowledge distillation loss using KL divergence.
#     """
#     return F.kl_div(
#         F.log_softmax(student_outputs / temperature, dim=-1),
#         F.softmax(teacher_outputs / temperature, dim=-1),
#         reduction='batchmean',
#         log_target=False
#     ) * (temperature ** 2)


class DistillationOfflineTrainer(OfflineTrainer):
	def __init__(self, cfg, env, agent, buffer, logger, teacher_model):
		super().__init__(cfg, env, agent, buffer, logger)
		self.teacher_model = teacher_model
		self.distillation_weight = cfg.distillation_weight
		self.distillation_temperature = cfg.distillation_temperature
		
		self.device = next(self.agent.model.parameters()).device
		self.pca = None
		self.pca_fitted = False
		self.pca_samples_threshold = 100
		self.teacher_latents = []
		self.scheduler = torch.optim.lr_scheduler.StepLR(self.agent.optim, step_size=1000, gamma=0.99)

		self.teacher_projection = nn.Linear(1376, 128).to(self.device)  # XM > 1M latent space, 1376 > 128
		
		# Initialize the projection layer
		nn.init.xavier_uniform_(self.teacher_projection.weight)
		nn.init.zeros_(self.teacher_projection.bias)
		
	def buffer_ready_for_pca(self):
		return self.buffer._num_eps >= self.pca_samples_threshold

	def fit_pca(self):
		print(f"Fitting PCA with {self.pca_samples_threshold} samples...")
		teacher_latents = []
		for _ in range(self.pca_samples_threshold):
			batch = self.buffer.sample()
			obs, _, _, task = batch
			with torch.no_grad():
				teacher_z = self.teacher_model.model.encode(obs[0].to(self.device), task)
			teacher_latents.append(teacher_z.cpu().numpy())
		
		self.pca = PCA(n_components=128)
		self.pca.fit(np.vstack(teacher_latents))
		self.pca_linear = nn.Linear(1376, 128, bias=False)
		self.pca_linear.weight.data = torch.tensor(self.pca.components_, dtype=torch.float32).to(self.device)
		self.pca_fitted = True
		print("PCA fitted successfully")

	def project_teacher_latent(self, teacher_z):
		if self.pca_fitted:
			projected = self.pca_linear(teacher_z)
		else:
			projected = self.teacher_projection(teacher_z)
		return projected

	def update_agent(self):
		batch = self.buffer.sample()
		obs, action, reward, task = batch 

		# Get teacher outputs
		with torch.no_grad():
			teacher_z = self.teacher_model.model.encode(obs[0].to(self.device), task)
			teacher_next_z = self.teacher_model.model.next(teacher_z, action[0].to(self.device), task)
			teacher_next_z_projected = self.project_teacher_latent(teacher_next_z)
			teacher_reward = self.teacher_model.model.reward(teacher_z, action[0].to(self.device), task)
			teacher_q = self.teacher_model.model.Q(teacher_z, action[0].to(self.device), task)

		# Get student outputs
		student_z = self.agent.model.encode(obs[0].to(self.device), task)
		student_next_z = self.agent.model.next(student_z, action[0].to(self.device), task)
		student_reward = self.agent.model.reward(student_z, action[0].to(self.device), task)
		student_q = self.agent.model.Q(student_z, action[0].to(self.device), task)

		# Compute distillation losses
		next_z_dist_loss = F.mse_loss(student_next_z, teacher_next_z_projected)
		reward_dist_loss = F.mse_loss(student_reward, teacher_reward)

		q_dist_loss = F.mse_loss(student_q, teacher_q)
		# KL divergence for Q-value distillation
		# q_dist_loss = F.kl_div(
		# 	F.log_softmax(student_q / self.distillation_temperature, dim=-1),
		# 	F.softmax(teacher_q / self.distillation_temperature, dim=-1),
		# 	reduction='batchmean'
		# ) * (self.distillation_temperature ** 2)

		# Compute auxiliary losses
		contrastive_loss = self.contrastive_loss(student_z, teacher_next_z_projected)
		consistency_loss = self.consistency_loss(student_z, action[0], task)

		# Compute original TD-MPC2 loss
		original_loss_dict = self.agent.update(obs, action, reward, task)
		
		# Combine losses
		dist_loss = next_z_dist_loss + reward_dist_loss + q_dist_loss
		aux_loss = contrastive_loss + consistency_loss
		total_loss = original_loss_dict['total_loss'] + self.distillation_weight * (dist_loss + aux_loss)
  
		## TODO: add
		# self.scheduler = torch.optim.lr_scheduler.StepLR(self.agent.optim, step_size=1000, gamma=0.99)
  
		dist_loss = (
			1.0 * next_z_dist_loss + 
			0.5 * reward_dist_loss + 
			0.6 * q_dist_loss
		)
		aux_loss = 0.5 * contrastive_loss + 0.5 * consistency_loss
		total_loss = original_loss_dict['total_loss'] + self.distillation_weight * (dist_loss + aux_loss)

		# Backward pass
		self.agent.optim.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.agent.cfg.grad_clip_norm)
		self.agent.optim.step()
		self.scheduler.step()

		# Update the original loss dictionary with the distillation and auxiliary losses
		original_loss_dict.update({
			'distillation_loss': dist_loss.item(),
			'auxiliary_loss': aux_loss.item(),
			'total_loss': total_loss.item()
		})

		# Log the losses to wandb
		wandb.log({
			'distillation_loss': dist_loss.item(),
			'auxiliary_loss': aux_loss.item(),
			'total_loss': total_loss.item(),
			# **aux_losses
		})

		return original_loss_dict

	def compute_auxiliary_losses(self, student_z, teacher_z, action, task):
		# Implement auxiliary losses here
		aux_losses = {}

		# 1. Contrastive loss
		contrastive_loss = self.contrastive_loss(student_z, teacher_z)
		aux_losses['contrastive_loss'] = contrastive_loss

		# 2. Consistency loss
		consistency_loss = self.consistency_loss(student_z, action, task)
		aux_losses['consistency_loss'] = consistency_loss

		# # 3. Reconstruction loss
		# reconstruction_loss = self.reconstruction_loss(student_z, self.agent.model.encode)
		# aux_losses['reconstruction_loss'] = reconstruction_loss

		return aux_losses

	def contrastive_loss(self, student_z, teacher_z, temperature=0.5):
		student_z = F.normalize(student_z, dim=-1)
		teacher_z = F.normalize(teacher_z, dim=-1)
		
		logits = torch.matmul(student_z, teacher_z.T) / temperature
		labels = torch.arange(student_z.shape[0], device=self.device)
		
		loss = F.cross_entropy(logits, labels)
		return loss

	def consistency_loss(self, z, action, task):
		next_z_pred = self.agent.model.next(z, action, task)
		next_obs, _, _, _ = self.buffer.sample()
		next_z_actual = self.agent.model.encode(next_obs[0].to(self.device), task)
		
		pred_delta = next_z_pred - z
		actual_delta = next_z_actual - z
		
		return F.mse_loss(pred_delta, actual_delta.detach())

	# def reconstruction_loss(self, z, encoder):
	# 	# Implement reconstruction loss
	# 	reconstructed_z = encoder(self.agent.model.decode(z))
	# 	return F.mse_loss(z, reconstructed_z)