import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
import warnings
warnings.filterwarnings('ignore')
import torch
import hydra
from termcolor import colored
from omegaconf import OmegaConf
from pathlib import Path

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_envs
from tdmpc2 import TDMPC2
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger, print_run
import multiprocessing

torch.backends.cudnn.benchmark = True


class DistillationTrainer(OnlineTrainer):
	"""
	Extended trainer that adds knowledge distillation from a teacher model.
	"""
	def __init__(self, cfg, env, eval_env, agent, teacher_agent, buffer, logger):
		super().__init__(cfg, env, eval_env, agent, buffer, logger)
		self.teacher_agent = teacher_agent
		self.teacher_agent.model.eval()  # Teacher in eval mode
		# Freeze teacher parameters
		for param in self.teacher_agent.model.parameters():
			param.requires_grad = False
		self.d_coef = cfg.get('d_coef', 0.4)  # Distillation coefficient
		
	def _compute_distillation_loss(self, obs, action):
		"""
		Compute MSE loss between student and teacher reward predictions.
		"""
		with torch.no_grad():
			# Get teacher's reward prediction
			teacher_reward = self.teacher_agent.model.reward(obs, action)
		
		# Get student's reward prediction
		student_reward = self.agent.model.reward(obs, action)
		
		# MSE loss
		distill_loss = torch.nn.functional.mse_loss(student_reward, teacher_reward)
		return distill_loss
	
	def _update(self, replay_iter):
		"""
		Override update to include distillation loss.
		"""
		# Sample batch from replay buffer
		batch = next(replay_iter)
		
		# Move to device
		obs = batch['obs'].to(self.cfg.device)
		action = batch['action'].to(self.cfg.device)
		reward = batch['reward'].to(self.cfg.device)
		next_obs = batch['next_obs'].to(self.cfg.device)
		
		# Get original TD-MPC2 losses
		metrics = {}
		
		# Compute model prediction
		z = self.agent.model.encode(obs)
		next_z = self.agent.model.encode(next_obs)
		
		# Latent dynamics (consistency loss)
		pred_next_z = self.agent.model.latent(z, action)
		consistency_loss = torch.nn.functional.mse_loss(pred_next_z, next_z)
		
		# Reward prediction loss
		pred_reward = self.agent.model.reward(z, action)
		reward_loss = torch.nn.functional.mse_loss(pred_reward, reward)
		
		# Value prediction (if using)
		if hasattr(self.agent.model, 'value'):
			pred_value = self.agent.model.value(z)
			# Compute TD target
			with torch.no_grad():
				next_value = self.agent.model.value(next_z)
				target_value = reward + self.cfg.discount * next_value
			value_loss = torch.nn.functional.mse_loss(pred_value, target_value)
		else:
			value_loss = torch.tensor(0.0)
		
		# Original TD-MPC2 loss (weighted combination)
		original_loss = (
			self.cfg.get('consistency_coef', 20) * consistency_loss +
			self.cfg.get('reward_coef', 0.1) * reward_loss +
			self.cfg.get('value_coef', 0.1) * value_loss
		)
		
		# Distillation loss
		distill_loss = self._compute_distillation_loss(z, action)
		
		# Total loss
		total_loss = original_loss + self.d_coef * distill_loss
		
		# Optimization step
		self.agent.model.optimizer.zero_grad()
		total_loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.agent.model.parameters(), 
			self.cfg.get('grad_clip_norm', 10.0)
		)
		self.agent.model.optimizer.step()
		
		# Logging
		metrics['consistency_loss'] = consistency_loss.item()
		metrics['reward_loss'] = reward_loss.item()
		metrics['value_loss'] = value_loss.item()
		metrics['distill_loss'] = distill_loss.item()
		metrics['total_loss'] = total_loss.item()
		
		return metrics


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Train single-task TD-MPC2 agent with knowledge distillation.
	
	Example usage:
		python train_distill.py \
			task=PickCube-v1 \
			model_size=1 \
			teacher_model_size=5 \
			teacher_checkpoint=path/to/teacher.pt \
			d_coef=0.4 \
			steps=1_000_000
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	
	# Parse config
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	
	# Ensure single-task
	assert not cfg.multitask, colored(
		'Multi-task not supported. Use single task with env_id parameter.',
		'red', attrs=['bold']
	)
	
	# Check for teacher checkpoint
	if not hasattr(cfg, 'teacher_checkpoint') or cfg.teacher_checkpoint is None:
		raise ValueError(
			"Must provide teacher_checkpoint path for distillation training. "
			"Example: teacher_checkpoint=checkpoints/teacher_pickcube.pt"
		)
	
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)
	print(colored('Teacher checkpoint:', 'cyan', attrs=['bold']), cfg.teacher_checkpoint)
	print(colored('Distillation coefficient (d_coef):', 'cyan', attrs=['bold']), cfg.get('d_coef', 0.4))
	
	# Initialize logger
	manager = multiprocessing.Manager()
	video_path = cfg.work_dir / 'eval_video'
	if cfg.save_video_local:
		os.makedirs(video_path, exist_ok=True)
	logger = Logger(cfg, manager)
	
	# Create environments
	env = make_envs(cfg, cfg.num_envs)
	eval_env = make_envs(
		cfg, 
		cfg.num_eval_envs, 
		video_path=video_path, 
		is_eval=True, 
		logger=logger
	)
	
	print_run(cfg)
	
	# Initialize student agent
	student_cfg = cfg.copy()
	student_cfg.model_size = cfg.model_size  
	student_agent = TDMPC2(student_cfg)
	print(colored(
		f'Student model size: {cfg.model_size}M parameters',
		'green', attrs=['bold']
	))
	
	# Initialize teacher agent
	teacher_cfg = cfg.copy()
	teacher_cfg.model_size = cfg.get('teacher_model_size', 5)  # Larger model
	teacher_agent = TDMPC2(teacher_cfg)
	
	# Load teacher checkpoint
	teacher_checkpoint_path = Path(cfg.teacher_checkpoint)
	if not teacher_checkpoint_path.exists():
		raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")
	
	print(colored('Loading teacher checkpoint...', 'yellow'))
	teacher_state = torch.load(teacher_checkpoint_path, map_location=cfg.device)
	teacher_agent.model.load_state_dict(teacher_state['model'])
	print(colored(
		f'Teacher model loaded: {cfg.get("teacher_model_size", 5)}M parameters',
		'green', attrs=['bold']
	))
	
	# Update wandb config
	if logger._wandb is not None:
		wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
		wandb_cfg['distillation'] = True
		wandb_cfg['d_coef'] = cfg.get('d_coef', 0.4)
		wandb_cfg['teacher_model_size'] = cfg.get('teacher_model_size', 5)
		logger._wandb.config.update(wandb_cfg, allow_val_change=True)
	
	# Create distillation trainer
	trainer = DistillationTrainer(
		cfg=cfg,
		env=env,
		eval_env=eval_env,
		agent=student_agent,
		teacher_agent=teacher_agent,
		buffer=Buffer(cfg),
		logger=logger,
	)
	
	# Train with distillation
	trainer.train()
	print('\nDistillation training completed successfully')
	
	# Save final student model
	final_checkpoint = 'tdmpc2/models/student_final.pt'
	torch.save({
		'model': student_agent.model.state_dict(),
		'cfg': cfg,
	}, final_checkpoint)
	print(colored(f'Student model saved to: {final_checkpoint}', 'green', attrs=['bold']))


if __name__ == '__main__':
	train()