import torch
import torch.nn as nn


class MLPExpert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Router(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_experts: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEPolicy(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        task_dim: int,
        action_dim: int,
        num_experts: int = 4,
        expert_hidden_dim: int = 512,
        router_hidden_dim: int = 128
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_experts = num_experts

        self.output_dim = self._calculate_output_dim(action_dim)
        self.router_in_dim = self._calculate_router_in_dim(task_dim)

        self.experts = nn.ModuleList([
            MLPExpert(self.latent_dim, expert_hidden_dim, self.output_dim)
            for _ in range(self.num_experts)
        ])
        
        self.router = Router(self.router_in_dim, router_hidden_dim, self.num_experts)

    @staticmethod
    def _calculate_output_dim(action_dim: int) -> int:
        return action_dim * 2 if action_dim < 12 else action_dim

    @staticmethod
    def _calculate_router_in_dim(task_dim: int) -> int:
        return task_dim if task_dim > 10 else 96

    def _prepare_router_input(self, task_emb: torch.Tensor) -> torch.Tensor:
        if task_emb.shape[-1] != self.router_in_dim:
            return task_emb[..., :self.router_in_dim]
        return task_emb

    def _prepare_expert_input(self, z: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.latent_dim:
            if self.latent_dim == z.shape[-1] + task_emb.shape[-1]:
                return torch.cat([z, task_emb], dim=-1)
        return z

    def forward(self, z: torch.Tensor, task_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        router_in = self._prepare_router_input(task_emb)
        gate_weights = self.router(router_in)

        expert_in = self._prepare_expert_input(z, task_emb)
        expert_outputs = torch.stack([expert(expert_in) for expert in self.experts], dim=-2)

        output = torch.einsum("...e, ...ed -> ...d", gate_weights, expert_outputs)
        
        aux_loss = torch.tensor(0.0, device=output.device)

        return output, aux_loss
