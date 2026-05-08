import torch
import torch.nn as nn
from omegaconf import OmegaConf
from common import layers


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda")

        self._setup_config()
        self._build_networks()
        self._build_action_masks()

        self.apply(self._init_weights)
        self.to(self.device)

    def _setup_config(self) -> None:
        self.is_moe_student = getattr(self.cfg, "is_moe_student", False)
        
        if self.is_moe_student:
            try:
                OmegaConf.set_struct(self.cfg, False)
            except Exception:
                pass
            self.cfg.latent_dim = 1376
            self.cfg.mlp_dim = 4096
            self.cfg.task_dim = 96

        self.task_dim = getattr(self.cfg, "task_dim", 96)

    def _build_networks(self) -> None:
        in_dim = self.cfg.latent_dim + self.task_dim + self.cfg.action_dim
        mlp_dims = [4096, 4096]

        self._encoder = layers.enc(self.cfg)
        self._dynamics = layers.mlp(in_dim, mlp_dims, self.cfg.latent_dim)
        self._reward = layers.mlp(in_dim, mlp_dims, self.cfg.num_bins)

        self._qs = nn.ModuleList([
            layers.mlp(in_dim, mlp_dims, self.cfg.num_bins)
            for _ in range(self.cfg.num_q)
        ])

        self._task_emb = nn.Embedding(len(self.cfg.tasks), self.task_dim)
        self._pi = self._create_policy(mlp_dims)

    def _create_policy(self, mlp_dims: list[int]) -> nn.Module:
        if self.is_moe_student:
            from common.moe import MoEPolicy
            return MoEPolicy(
                latent_dim=self.cfg.latent_dim + self.task_dim,
                task_dim=self.task_dim,
                action_dim=self.cfg.action_dim,
                num_experts=getattr(self.cfg, "num_experts", 4)
            )

        return layers.mlp(
            self.cfg.latent_dim + self.task_dim,
            mlp_dims,
            2 * self.cfg.action_dim
        )

    def _build_action_masks(self) -> None:
        num_tasks = len(self.cfg.tasks)
        self._action_masks = torch.zeros(num_tasks, self.cfg.action_dim, device=self.device)

        for i in range(num_tasks):
            action_dim = (
                self.cfg.action_dims[i]
                if hasattr(self.cfg, "action_dims")
                else self.cfg.action_dim
            )
            self._action_masks[i, :action_dim] = 1.0

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, "data") and m.bias is not None:
                m.bias.data.fill_(0.0)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight.data, -0.1, 0.1)

    def _expand_task_embedding(self, z: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        task_emb = self._task_emb(task.long())
        if z.ndim > task_emb.ndim:
            task_emb = task_emb.unsqueeze(0).expand(*z.shape[:-1], -1)
        return task_emb

    def _prepare_obs_task_input(self, obs: torch.Tensor | dict, task: torch.Tensor) -> torch.Tensor:
        obs_tensor = obs[getattr(self.cfg, "obs", "state")] if isinstance(obs, dict) else obs
        task_emb = self._task_emb(task.long())

        if obs_tensor.ndim > task_emb.ndim:
            task_emb = task_emb.unsqueeze(0).expand(*obs_tensor.shape[:-1], -1)

        return torch.cat([obs_tensor, task_emb], dim=-1)

    def encode(self, obs: torch.Tensor | dict, task: torch.Tensor) -> torch.Tensor:
        model_input = self._prepare_obs_task_input(obs, task)

        if isinstance(self._encoder, nn.ModuleDict):
            obs_key = getattr(self.cfg, "obs", "state")
            return self._encoder[obs_key](model_input)

        return self._encoder(model_input)

    def next(self, z: torch.Tensor, action: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        task_emb = self._expand_task_embedding(z, task)
        model_input = torch.cat([z, task_emb, action], dim=-1)
        return self._dynamics(model_input)

    def reward(self, z: torch.Tensor, action: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        task_emb = self._expand_task_embedding(z, task)
        model_input = torch.cat([z, task_emb, action], dim=-1)
        return self._reward(model_input)

    def pi(self, z: torch.Tensor, task: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task_emb = self._expand_task_embedding(z, task)

        if self.is_moe_student:
            pi_out, _ = self._pi(z, task_emb)
            return pi_out.chunk(2, dim=-1)

        model_input = torch.cat([z, task_emb], dim=-1)
        return self._pi(model_input).chunk(2, dim=-1)

    def Q(self, z: torch.Tensor, action: torch.Tensor, task: torch.Tensor, return_all: bool = False) -> torch.Tensor:
        mask = self._action_masks[task.long()]
        masked_action = action * mask

        task_emb = self._expand_task_embedding(z, task)
        model_input = torch.cat([z, task_emb, masked_action], dim=-1)

        qs = torch.stack([q_net(model_input) for q_net in self._qs], dim=0)

        if return_all:
            return qs
        return torch.min(qs, dim=0)[0]

    def track_q_grad(self, mode: bool = True) -> None:
        for p in self._qs.parameters():
            p.requires_grad = mode
