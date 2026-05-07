import numpy as np
import torch
import wandb
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from utils import get_distillation_coefficient

class TDMPC2:
    """
    TD-MPC2 agent. Implements training + inference.
    Supports MoE Student architecture for MT30 Distillation.
    """

    def __init__(self, cfg, teacher_model=None):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.model = WorldModel(cfg).to(self.device)
        self.teacher_model = teacher_model
        
        # Inject MoE Policy if configured as an MoE student
        if getattr(self.cfg, 'is_moe_student', False):
            from common.moe import MoEPolicy
            # Assuming task_dim is present for MT30
            self.model._pi = MoEPolicy(cfg.latent_dim, cfg.task_dim, cfg.action_dim, num_experts=4).to(self.device)

        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
        ], lr=self.cfg.lr)
        
        # Policy optimizer catches MoE parameters (experts + router) automatically
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
        
        self.model.eval()
        self.scale = RunningScale(cfg)
        self.cfg.iterations += 2*int(cfg.action_dim >= 20) 
        
        if self.cfg.multitask:
            self.discount = torch.tensor(
                [self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda'
            )
        else:
            self.discount = self._get_discount(cfg.episode_length)

    def _get_discount(self, episode_length):
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        
        def dequantize_tensor(tensor):
            if tensor.is_quantized:
                return tensor.dequantize()
            return tensor

        dequantized_state_dict = {
            k: dequantize_tensor(v) if isinstance(v, torch.Tensor) else v
            for k, v in state_dict["model"].items()
        }    
        self.model.load_state_dict(dequantized_state_dict, strict=False) 

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        obs = obs.to(self.device, non_blocking=True).unsqueeze(0)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        
        z = self.model.encode(obs, task)
        
        if self.cfg.mpc:
            a = self.plan(z, t0=t0, eval_mode=eval_mode, task=task)
        else:
            # Handle standard policy vs MoE policy output
            pi_out = self.model.pi(z, task)
            a = pi_out[int(not eval_mode)][0] if isinstance(pi_out, tuple) else pi_out[0]
            
        return a.cpu()

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        G, discount = 0, 1
        for t in range(self.cfg.horizon):
            reward = math.two_hot_inv(self.model.reward(z, actions[t], task), self.cfg)
            z = self.model.next(z, actions[t], task)
            G += discount * reward
            discount *= self.discount[torch.tensor(task)] if self.cfg.multitask else self.discount
        
        pi_out = self.model.pi(z, task)
        pi_action = pi_out[1] if isinstance(pi_out, tuple) else pi_out[0]
        return G + discount * self.model.Q(z, pi_action, task, return_type='avg')

    @torch.no_grad()
    def plan(self, z, t0=False, eval_mode=False, task=None):
        if self.cfg.num_pi_trajs > 0:
            pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
            _z = z.repeat(self.cfg.num_pi_trajs, 1)
            for t in range(self.cfg.horizon-1):
                pi_out = self.model.pi(_z, task)
                pi_actions[t] = pi_out[1] if isinstance(pi_out, tuple) else pi_out[0]
                _z = self.model.next(_z, pi_actions[t], task)
            pi_out = self.model.pi(_z, task)
            pi_actions[-1] = pi_out[1] if isinstance(pi_out, tuple) else pi_out[0]

        z = z.repeat(self.cfg.num_samples, 1)
        mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(self.cfg.horizon, self.cfg.action_dim, device=self.device)
        if not t0 and hasattr(self, '_prev_mean'):
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
        
        pi_out = self.model.pi(zs, task)
        # Handle standard tuple vs MoE tuple
        if len(pi_out) == 4:
            _, pis, log_pis, _ = pi_out
        else:
            pis, _ = pi_out
            log_pis = torch.zeros_like(pis[..., 0]) # Placeholder if MoE doesn't output log_prob yet
            
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
        pi_out = self.model.pi(next_z, task)
        pi = pi_out[1] if isinstance(pi_out, tuple) and len(pi_out) == 4 else pi_out[0]
        discount = self.discount[task].unsqueeze(-1) if self.cfg.multitask else self.discount
        return reward + discount * self.model.Q(next_z, pi, task, return_type='min', target=True)

    def update(self, buffer, step: int = 0):
        obs, action, reward, task = buffer.sample()
    
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
            reward_loss += math.soft_ce(reward_preds[t], reward[t], self.cfg).mean() * self.cfg.rho**t
            for q in range(self.cfg.num_q):
                value_loss += math.soft_ce(qs[q][t], td_targets[t], self.cfg).mean() * self.cfg.rho**t
        
        consistency_loss *= (1/self.cfg.horizon)
        reward_loss *= (1/self.cfg.horizon)
        value_loss *= (1/(self.cfg.horizon * self.cfg.num_q))
  
        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss
        )
  
        # MT30 MoE Knowledge Distillation (Policy & Latent Alignment)
        if self.teacher_model is not None:
            alpha = get_distillation_coefficient(
                step=step,
                schedule=self.cfg.distillation.schedule,
                total_steps=self.cfg.steps,
                base_coef=self.cfg.distillation.d_coef
            )

            # Get Teacher Targets
            with torch.no_grad():
                teacher_z = self.teacher_model.model.encode(obs[0], task)
                t_pi_out = self.teacher_model.model.pi(teacher_z, task)
                teacher_pi = t_pi_out[1] if len(t_pi_out) == 4 else t_pi_out[0]

            # Get Student Predictions
            student_z = zs[0] # From earlier encode
            s_pi_out = self.model.pi(student_z, task)
            student_pi = s_pi_out[1] if len(s_pi_out) == 4 else s_pi_out[0]

            # Distillation Objectives
            policy_distill_loss = F.mse_loss(student_pi, teacher_pi)
            latent_distill_loss = F.mse_loss(student_z, teacher_z)
            
            # Combine Distillation Loss (alpha applied dynamically)
            distill_loss = policy_distill_loss + self.cfg.get('latent_distill_alpha', 0.5) * latent_distill_loss
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
                "policy_distill_loss": float(policy_distill_loss.mean().item()),
                "latent_distill_loss": float(latent_distill_loss.mean().item())
            })
   
        if wandb.run is not None:
            wandb.log(stats)
        return stats