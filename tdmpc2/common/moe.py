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
    Experts are specialized in task clusters from MT30.
    """
    def __init__(self, latent_dim, task_dim, action_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            MLPExpert(latent_dim, 512, action_dim) for _ in range(num_experts)
        ])
        
        # The router uses task embeddings to select experts
        self.router = nn.Sequential(
            nn.Linear(task_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, z, task_emb):
        # Compute expert weights based on task context
        gate_weights = self.router(task_emb) # [batch, num_experts]
        
        # Stack expert outputs: [batch, num_experts, action_dim]
        expert_outputs = torch.stack([expert(z) for expert in self.experts], dim=1)
        
        # Weighted sum of expert outputs
        output = torch.einsum('be,bed->bd', gate_weights, expert_outputs)
        return output, gate_weights