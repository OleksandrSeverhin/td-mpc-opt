import math
import numpy as np
import torch
import wandb
import torch.nn.functional as F

from common import math as common_math
from common.scale import RunningScale
from common.world_model import WorldModel

class TDMPC2:
    """
    TD-MPC2 agent. Implements training + inference.
    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg, teacher_model=None):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.model = WorldModel(cfg).to(self.device)
        self.teacher_model = teacher_model
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
        ], lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20)
        self.discount = torch.tensor(
            [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
        ) if self.cfg.multitask else self._get_discount(cfg.episode_length)

    def _get_discount(self, episode_length):
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def distill(self, obs, action, task=None):
        z = self.world_model.encode(obs, task)
        z_next, r = self.world_model.step(z, action, task)
        return z_next, r

    def save(self, fp):
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        state_dict = fp if isinstance(fp, dict) else torch.load(fp, weights_only=False)
        
        def dequantize_tensor(tensor):
            if tensor.is_quantized:
                return tensor.dequantize()
            return tensor

        dequantized_state_dict = {
            k: dequantize_tensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in state_dict["model"].items()
        }    

        self.model.load_state_dict(dequantized_state_dict)

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        
        # ManiSkill3 Task Fallback Bridge
        if task is None:
            task = torch.zeros(obs.shape[0], dtype=torch.long, device=self.device)
        elif task is not None and not isinstance(task, torch.Tensor):
            task = torch.tensor([task], device=self.device)
            
        z = self.model.encode(obs, task)
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            a = self.model.pi(z, task)[int(not eval_mode)][0]
            
        # Removed .cpu() to keep tensors on GPU for ManiSkill3 SAPIEN engine
        return a

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = common_math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G += discount * reward
            discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
        return G + discount * self.model.Q(z, self.model.pi(z, task)[1], task, return_type='avg')

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None):
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_actions[t] = self.model.pi(_z, task)[1]
                _z = self.model.next(_z, pi_actions[t], task)
            pi_actions[-1] = self.model.pi(_z, task)[1]

        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
        if self.cfg.num_pi_trajs > 0:
            actions[:, :self.cfg.num_pi_trajs] = pi_actions
    
        for _ in range(self.cfg.iterations):
            actions[:, self.cfg.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)) \
                .clamp(-1, 1)
            if self.cfg.multitask:
                actions = actions * self.model._action_masks[task]

            value = self._estimate_value(z, actions, task).nan_to_num_(0)
            elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9)) \
                .clamp_(self.cfg.min_std, self.cfg.max_std)
            if self.cfg.multitask:
                mean = mean * self.model._action_masks[task]
                std = std * self.model._action_masks[task]

        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        a, std = actions[0], std[0]
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a.clamp_(-1, 1)
        
    def update_pi(self, zs, task):
        self.pi_optim.zero_grad(set_to_none=True)
        self.model.track_q_grad(False)
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg')
        self.scale.update(qs[0])
        qs = self.scale(qs)

        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.model.track_q_grad(True)

        return pi_loss.item()
 
    @torch.no_grad()
    def _td_target(self, next_z, reward, task):
        pi = self.model.pi(next_z, task)[1]
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

    def update(self, buffer, step: int = 0):
        obs, action, reward, task = buffer.sample()
        
        # ManiSkill3 Bridge: Assign default task ID for MT30 Teacher Compatibility
        if task is None:
            task = torch.zeros(obs.shape[1], dtype=torch.long, device=self.device)
    
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            td_targets = self._td_target(next_z, reward, task)
   
        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
        z = self.model.encode(obs[0], task)
        zs[0] = z
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t
            zs[t+1] = z

        _zs = zs[:-1]
        qs = self.model.Q(_zs, action, task, return_type='all')
        reward_preds = self.model.reward(_zs, action, task)
        
        reward_loss, value_loss = 0, 0
        for t in range(self.cfg.horizon):
            reward_loss += common_math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += common_math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
        consistency_loss *= (1/self.cfg.horizon)
        reward_loss *= (1/self.cfg.horizon)
        value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
  
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )
  
        if self.teacher_model is not None:
            # Inline Dynamic Coefficient Calculation
            if self.cfg.distillation.schedule == 'constant':
                alpha = self.cfg.distillation.d_coef
            elif self.cfg.distillation.schedule == 'linear':
                alpha = self.cfg.distillation.d_coef * max(1.0 - (step / self.cfg.steps), 0.1)
            elif self.cfg.distillation.schedule == 'cosine':
                alpha = self.cfg.distillation.d_coef * 0.5 * (1.0 + math.cos(math.pi * step / self.cfg.steps))
            else:
                alpha = self.cfg.distillation.d_coef

            with torch.no_grad():
                # ManiSkill3 Bridge: Slice 35-dim obs to 24-dim and 8-dim action to 6-dim for Teacher
                t_obs = obs[0][:, :24] if obs[0].shape[-1] >= 24 else obs[0]
                t_act = action[0][:, :6] if action[0].shape[-1] >= 6 else action[0]
                
                teacher_z = self.teacher_model.model.encode(t_obs, task)
                teacher_reward_logits = self.teacher_model.model.reward(teacher_z, t_act, task)
                teacher_reward = common_math.two_hot_inv(teacher_reward_logits, self.cfg)

            student_z = self.model.encode(obs[0], task)
            student_reward_logits = self.model.reward(student_z, action[0], task)
            student_reward = common_math.two_hot_inv(student_reward_logits, self.cfg)

            reward_distill_loss = F.mse_loss(student_reward, teacher_reward)
            distill_loss = reward_distill_loss
            total_loss = total_loss + alpha * distill_loss

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()

        pi_loss = self.update_pi(zs.detach(), task)
        self.model.soft_update_target_Q()

        self.model.eval()
        stats = {
            "consistency_loss": float(consistency_loss.mean().item()),
            "reward_loss": float(reward_loss.mean().item()),
            "value_loss": float(value_loss.mean().item()),
            "pi_loss": pi_loss,
            "total_loss": float(total_loss.mean().item()),
            "grad_norm": float(grad_norm),
            "pi_scale": float(self.scale.value),
            "distillation_coef": float(alpha) if self.teacher_model is not None else 0.0,
        }
        if self.teacher_model is not None:
            stats.update({
                "distillation_loss": float(distill_loss.mean().item()),
                "reward_distill_loss": float(reward_distill_loss.mean().item()),
            })
   
        if wandb.run is not None:
            wandb.log(stats)
        return stats