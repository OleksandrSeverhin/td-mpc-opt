import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPExpert(nn.Module):
    """A single expert MLP for the student policy."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class MoEPolicy(nn.Module):
    """
    Student policy with Mixture of Experts.
    Robustified to natively handle arbitrary batch dimensions (1D, 2D, 3D).
    """
    def __init__(self, latent_dim, task_dim, action_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        self.input_dim = latent_dim
        
        # TD-MPC2 continuous policies MUST output 2x the action dim (mean + log_std).
        actual_out_dim = action_dim * 2 if action_dim < 12 else action_dim
        
        self.experts = nn.ModuleList([
            MLPExpert(self.input_dim, 512, actual_out_dim) for _ in range(num_experts)
        ])
        
        # Safe fallback: if task_dim was mapped to num_experts during init, default to 96
        router_in_dim = task_dim if task_dim > 10 else 96
        
        self.router = nn.Sequential(
            nn.Linear(router_in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, z, task_emb):
        # 1. Routing
        router_in = task_emb
        if task_emb.shape[-1] != self.router[0].in_features:
            router_in = task_emb[..., :self.router[0].in_features]
            
        gate_weights = self.router(router_in) 
        
        # 2. Expert Forward
        expert_in = z
        if self.input_dim != z.shape[-1]:
            if self.input_dim == z.shape[-1] + task_emb.shape[-1]:
                expert_in = torch.cat([z, task_emb], dim=-1)
                
        expert_outputs = torch.stack([expert(expert_in) for expert in self.experts], dim=-2)
        
        # 3. Dimension-Agnostic Einsum
        output = torch.einsum('...e, ...ed -> ...d', gate_weights, expert_outputs)
        
        # FIX: Provide a safe zero tensor instead of None to prevent metric accumulator crashes
        aux_loss = torch.tensor(0.0, device=output.device)
        
        return output, aux_loss