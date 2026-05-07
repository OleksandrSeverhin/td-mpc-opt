import torch
import torch.nn as nn
from omegaconf import OmegaConf
from common import layers

class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda')

        if getattr(cfg, 'is_moe_student', False):
            try:
                OmegaConf.set_struct(self.cfg, False)
            except Exception:
                pass
            self.cfg.latent_dim = 1376
            self.cfg.mlp_dim = 4096
            self.cfg.task_dim = 96

        self.task_dim = getattr(cfg, 'task_dim', 96)
        
        in_dim = cfg.latent_dim + self.task_dim + cfg.action_dim
        mlp_dims = [4096, 4096]

        self._encoder = layers.enc(cfg)
        
        self._dynamics = layers.mlp(in_dim, mlp_dims, cfg.latent_dim)
        self._reward = layers.mlp(in_dim, mlp_dims, cfg.num_bins)
        self._Qs = nn.ModuleList([
            layers.mlp(in_dim, mlp_dims, cfg.num_bins)
            for _ in range(cfg.num_q)
        ])
        
        self._task_emb = nn.Embedding(len(cfg.tasks), self.task_dim)

        if getattr(cfg, 'is_moe_student', False):
            from common.moe import MoEPolicy
            self._pi = MoEPolicy(
                latent_dim=cfg.latent_dim + self.task_dim,
                task_dim=self.task_dim,
                action_dim=cfg.action_dim,
                num_experts=getattr(cfg, 'num_experts', 4)
            )
        else:
            self._pi = layers.mlp(cfg.latent_dim + self.task_dim, mlp_dims, 2 * cfg.action_dim)

        self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim, device=self.device)
        for i in range(len(cfg.tasks)):
            if hasattr(cfg, 'action_dims'):
                self._action_masks[i, :cfg.action_dims[i]] = 1.
            else:
                self._action_masks[i, :] = 1.

        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data') and m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight.data, -0.1, 0.1)

    def _get_expanded_task(self, z, task):
        task_emb = self._task_emb(task.long())
        if z.ndim > task_emb.ndim:
            task_emb = task_emb.unsqueeze(0).expand(*z.shape[:-1], -1)
        return task_emb

    def task_emb(self, obs, task):
        if isinstance(obs, dict):
            obs_key = getattr(self.cfg, 'obs', 'state')
            obs = obs[obs_key]
        task_emb = self._task_emb(task.long())
        if obs.ndim > task_emb.ndim:
            task_emb = task_emb.unsqueeze(0).expand(*obs.shape[:-1], -1)
        return torch.cat([obs, task_emb], dim=-1)

    def encode(self, obs, task):
        x = self.task_emb(obs, task)
        if isinstance(self._encoder, nn.ModuleDict):
            obs_key = getattr(self.cfg, 'obs', 'state')
            return self._encoder[obs_key](x)
        return self._encoder(x)

    def next(self, z, action, task):
        task_emb = self._get_expanded_task(z, task)
        x = torch.cat([z, task_emb, action], dim=-1)
        return self._dynamics(x)

    def reward(self, z, action, task):
        task_emb = self._get_expanded_task(z, task)
        x = torch.cat([z, task_emb, action], dim=-1)
        return self._reward(x)

    def pi(self, z, task):
        task_emb = self._get_expanded_task(z, task)
        
        if getattr(self.cfg, 'is_moe_student', False):
            pi_out, _ = self._pi(z, task_emb)
            return pi_out.chunk(2, dim=-1)
            
        x = torch.cat([z, task_emb], dim=-1)
        return self._pi(x).chunk(2, dim=-1)

    def Q(self, z, action, task, return_all=False):
        mask = self._action_masks[task.long()]
        action = action * mask
        
        task_emb = self._get_expanded_task(z, task)
        x = torch.cat([z, task_emb, action], dim=-1)
        
        qs = torch.stack([q_net(x) for q_net in self._Qs], dim=0)
        
        if return_all:
            return qs
        return torch.min(qs, dim=0)[0]

    def track_q_grad(self, mode=True):
        for p in self._Qs.parameters():
            p.requires_grad = mode