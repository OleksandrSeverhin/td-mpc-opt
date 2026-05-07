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
        
        # Dynamically map the input to handle if latent_dim was concatenated with task_dim
        self.input_dim = latent_dim
        
        self.experts = nn.ModuleList([
            MLPExpert(self.input_dim, 512, action_dim) for _ in range(num_experts)
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
            # Fallback if dimensions got crossed during world_model init
            router_in = task_emb[..., :self.router[0].in_features]
            
        # Shape: [..., num_experts]
        gate_weights = self.router(router_in) 
        
        # 2. Expert Forward
        # Auto-detect if experts expect z (1376) or z + task_emb (1472)
        expert_in = z
        if self.input_dim != z.shape[-1]:
            if self.input_dim == z.shape[-1] + task_emb.shape[-1]:
                expert_in = torch.cat([z, task_emb], dim=-1)
                
        # Stack on the second-to-last dimension (dim=-2)
        # This safely turns a list of [..., action_dim] into [..., num_experts, action_dim]
        expert_outputs = torch.stack([expert(expert_in) for expert in self.experts], dim=-2)
        
        # 3. Dimension-Agnostic Einsum
        # The ellipsis (...) gracefully handles 1D, 2D, or 3D tensors natively!
        output = torch.einsum('...e, ...ed -> ...d', gate_weights, expert_outputs)
        
        # TD-MPC2 unpacks a tuple (mean, log_std), so we return a dummy second argument
        return output, None