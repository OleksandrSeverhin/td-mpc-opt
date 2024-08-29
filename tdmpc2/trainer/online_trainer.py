from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict.tensordict import TensorDict

from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
		), batch_size=(1,))
		return td

	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, True, True
		while self._step <= self.cfg.steps:

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

				if self._step > 0:
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				action = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					# _train_metrics = self.agent.update(self.buffer)
					_train_metrics = self.update_agent()
				train_metrics.update(_train_metrics)

			self._step += 1
	
		self.logger.finish(self.agent)

def distillation_loss(student_outputs, teacher_outputs, temperature=2.0):
	"""
	Compute the knowledge distillation loss.
	"""
	return F.kl_div(
		F.log_softmax(student_outputs / temperature, dim=1),
		F.softmax(teacher_outputs / temperature, dim=1),
		reduction='batchmean'
	) * (temperature ** 2)


class DistillationOnlineTrainer(OnlineTrainer):
	def __init__(self, cfg, env, agent, buffer, logger, teacher_model):
		super().__init__(cfg, env, agent, buffer, logger)
		self.teacher_model = teacher_model
		self.distillation_weight = cfg.distillation_weight
		self.distillation_temperature = cfg.distillation_temperature
		
		self.device = next(self.agent.model.parameters()).device
		self.teacher_projection = nn.Linear(512, 128).to(self.device)  # 5M > 1M latent space, 512 > 128

	def update_agent(self):
		batch = self.buffer.sample()
		obs, action, reward, task = batch 

		# Get teacher outputs
		with torch.no_grad():
			teacher_z = self.teacher_model.model.encode(obs[0].to(self.device), task)
			teacher_next_z = self.teacher_model.model.next(teacher_z, action[0].to(self.device), task)
			teacher_next_z_projected = self.teacher_projection(teacher_next_z.to(self.device))
			teacher_reward = self.teacher_model.model.reward(teacher_z, action[0].to(self.device), task)

		# Get student outputs
		student_z = self.agent.model.encode(obs[0].to(self.device), task)
		student_next_z = self.agent.model.next(student_z, action[0].to(self.device), task)
		student_reward = self.agent.model.reward(student_z, action[0].to(self.device), task)

		# Compute distillation loss
		dist_loss = distillation_loss(student_next_z / self.distillation_temperature, 
										teacher_next_z_projected / self.distillation_temperature) + \
					distillation_loss(student_reward / self.distillation_temperature, 
										teacher_reward / self.distillation_temperature)

		# backward for distillation loss
		self.agent.optim.zero_grad()
		(self.distillation_weight * dist_loss).backward(retain_graph=True)

		# Compute original TD-MPC2 loss
		original_loss_dict = self.agent.update(obs, action, reward, task)
		
		# Optimize after both backward passes
		# grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.agent.cfg.grad_clip_norm)
		self.agent.optim.step()

		# Combine losses
		total_loss = original_loss_dict['total_loss'] + self.distillation_weight * dist_loss.item() # 0.5 weight
		original_loss_dict['total_loss'] = total_loss

		# self.logger.log('train/distillation_loss', dist_loss.item())
		# self.logger.log('train/total_loss', total_loss)

		return original_loss_dict