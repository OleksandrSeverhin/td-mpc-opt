import torch
import torch.nn.functional as F

class DistillTrainer:
    def __init__(self, student, teachers, alpha=0.5, lr=3e-4):
        self.student = student
        self.teachers = teachers 
        self.alpha = alpha
        self.optimizer = torch.optim.Adam(self.student.model.parameters(), lr=lr)

    def update(self, replay_buffer):
        # 1. Sample: [Horizon+1, Batch, Dim]
        obs_seq, action_seq, reward_seq, task_ids_seq = replay_buffer.sample()
        
        # 2. Extract t=0 and move to GPU
        obs = obs_seq[0].cuda()
        task_ids = task_ids_seq[0].cuda().long()
        if task_ids.dim() > 1:
            task_ids = task_ids.squeeze(-1)

        # 3. Vectorized Teacher Targets
        with torch.no_grad():
            # In MT30 distillation, we use one multi-task teacher for all samples
            teacher = self.teachers[0] 
            
            # Get teacher latent and policy mean for the whole batch
            z_t = teacher.model.encode(obs, task_ids)
            mu_t, _ = teacher.model.pi(z_t, task_ids)
            
            target_a = mu_t
            target_z = z_t

        # 4. Student Forward Pass
        z_s = self.student.model.encode(obs, task_ids)
        task_emb = self.student.model._task_emb(task_ids)
        
        # Call MoE Policy with full batch (256, 1376)
        pred_a_dist, gate_weights = self.student.model._pi(z_s, task_emb)
        
        # Split mu and log_std; we distill the mu (mean)
        pred_a = pred_a_dist.chunk(2, dim=-1)[0]

        # 5. Combined Loss
        policy_loss = F.mse_loss(pred_a, target_a)
        latent_loss = F.mse_loss(z_s, target_z)
        balance_loss = self._entropy_loss(gate_weights)
        
        total_loss = policy_loss + self.alpha * latent_loss + 0.01 * balance_loss
        
        # Optimize
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'latent_loss': latent_loss.item(),
            'balance_loss': balance_loss.item()
        }

    def _entropy_loss(self, weights):
        # Encourages expert specialization
        return -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()