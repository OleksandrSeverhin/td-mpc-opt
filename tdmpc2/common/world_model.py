import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from common import layers

class WorldModel(nn.Module):
    """
    TD-MPC2 World Model.
    Includes internal weight initialization and naming aligned with TDMPC2 wrapper.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda')
        
        # 1. Dimensions
        # task_dim: 352 for model_size 5, 448 for model_size 7.
        self.task_dim = 32 * math.ceil(16 * (cfg.model_size / 5)) 
        
        # 2. Network Components
        self._encoder = layers.enc(cfg)
        
        # Dynamics: (latent + action) -> latent
        self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.latent_dim)
        
        # Reward: (latent + action) -> num_bins
        self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.num_bins)
        
        # Q-ensemble: Manual implementation using ModuleList.
        # Naming must be _Qs to match the attribute lookup in tdmpc2.py.
        self._Qs = nn.ModuleList([
            layers.mlp(cfg.latent_dim + cfg.action_dim, cfg.mlp_dim, cfg.num_bins)
            for _ in range(cfg.num_q)
        ])
        
        # Task Embeddings
        self._task_emb = nn.Embedding(len(cfg.tasks), self.task_dim)
        
        # 3. Policy Initialization
        if getattr(cfg, 'is_moe_student', False):
            from common.layers import MoEPolicy
            self._pi = MoEPolicy(cfg.latent_dim + self.task_dim, cfg.action_dim, cfg)
        else:
            try:
                self._pi = layers.pi(cfg)
            except AttributeError:
                # Fallback to standard MLP if layers.pi is missing
                self._pi = layers.mlp(cfg.latent_dim, cfg.mlp_dim, 2 * cfg.action_dim)
            
        # 4. Action Masks
        self._action_masks = torch.zeros(len(cfg.tasks), cfg.action_dim, device=self.device)
        for i in range(len(cfg.tasks)):
            self._action_masks[i, :cfg.action_dims[i]] = 1.
            
        # Internal Weight Initialization
        self.apply(self._init_weights)
        self.to(self.device)

    def _init_weights(self, m):
        """Standard orthogonal initialization for linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data') and m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight.data, -0.1, 0.1)

    def task_emb(self, obs, task):
        """Retrieves task embeddings and concatenates them with observations."""
        if isinstance(obs, dict):
            obs = obs[self.cfg.obs]
        
        task = task.long()
        emb = self._task_emb(task)
        
        if obs.ndim != emb.ndim:
            emb = emb.reshape(obs.shape[:-1] + (self.task_dim,))
            
        return torch.cat([obs, emb], dim=-1)

    def encode(self, obs, task):
        """Encodes observation into latent state."""
        return self._encoder(self.task_emb(obs, task))

    def next(self, z, action):
        """Predicts next latent state."""
        return self._dynamics(torch.cat([z, action], dim=-1))

    def reward(self, z, action):
        """Predicts reward."""
        return self._reward(torch.cat([z, action], dim=-1))

    def pi(self, z, task):
        """Samples action from policy network."""
        if getattr(self.cfg, 'is_moe_student', False):
            task_emb = self._task_emb(task.long())
            pi_out, _ = self._pi(z, task_emb)
            return pi_out.chunk(2, dim=-1)
        return self._pi(z).chunk(2, dim=-1)

    def Q(self, z, action, task, return_all=False):
        """Predicts Q-values from the manual ensemble."""
        mask = self._action_masks[task.long()]
        action = action * mask
        x = torch.cat([z, action], dim=-1)
        
        # Stack predictions from each head in the ensemble
        qs = torch.stack([q_net(x) for q_net in self._Qs], dim=0)
        
        if return_all:
            return qs
        return torch.min(qs, dim=0)[0]

    def track_q_grad(self, mode=True):
        """Enables/Disables gradient tracking for Q-functions."""
        for p in self._Qs.parameters():
            p.requires_grad = mode