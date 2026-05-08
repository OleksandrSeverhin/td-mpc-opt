import torch
import torch.nn.functional as F
from torch.optim import Adam


class DistillTrainer:
    def __init__(self, student, teachers: dict | list, alpha: float = 0.5, lr: float = 3e-4, device: str = "cuda"):
        self.student = student
        self.teacher = teachers[0]
        self.alpha = alpha
        self.device = torch.device(device)
        self.optimizer = Adam(self.student.model.parameters(), lr=lr)

    def update(self, replay_buffer) -> dict[str, float]:
        obs_seq, _, _, task_ids_seq = replay_buffer.sample()
        
        obs, task_ids = self._prepare_inputs(obs_seq, task_ids_seq)

        target_a, target_z = self._get_teacher_targets(obs, task_ids)
        pred_a, z_s, gate_weights = self._get_student_predictions(obs, task_ids)

        return self._compute_and_optimize(pred_a, target_a, z_s, target_z, gate_weights)

    def _prepare_inputs(self, obs_seq: torch.Tensor, task_ids_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs = obs_seq[0].to(self.device)
        task_ids = task_ids_seq[0].to(self.device).long()
        
        if task_ids.dim() > 1:
            task_ids = task_ids.squeeze(-1)
            
        return obs, task_ids

    @torch.no_grad()
    def _get_teacher_targets(self, obs: torch.Tensor, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_t = self.teacher.model.encode(obs, task_ids)
        mu_t, _ = self.teacher.model.pi(z_t, task_ids)
        return mu_t, z_t

    def _get_student_predictions(self, obs: torch.Tensor, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_s = self.student.model.encode(obs, task_ids)
        task_emb = self.student.model._task_emb(task_ids)
        
        pred_a_dist, gate_weights = self.student.model._pi(z_s, task_emb)
        pred_a = pred_a_dist.chunk(2, dim=-1)[0]
        
        return pred_a, z_s, gate_weights

    def _compute_and_optimize(
        self,
        pred_a: torch.Tensor,
        target_a: torch.Tensor,
        z_s: torch.Tensor,
        target_z: torch.Tensor,
        gate_weights: torch.Tensor
    ) -> dict[str, float]:
        policy_loss = F.mse_loss(pred_a, target_a)
        latent_loss = F.mse_loss(z_s, target_z)
        balance_loss = self._entropy_loss(gate_weights)
        
        total_loss = policy_loss + self.alpha * latent_loss + 0.01 * balance_loss
        
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "latent_loss": latent_loss.item(),
            "balance_loss": balance_loss.item()
        }

    @staticmethod
    def _entropy_loss(weights: torch.Tensor) -> torch.Tensor:
        return -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
