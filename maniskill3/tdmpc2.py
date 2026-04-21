import torch
import torch.nn.functional as F
from common.world_model import WorldModel

class TDMPC2:
    def __init__(self, cfg, teacher=None):
        self.cfg = cfg
        self.device = torch.device('cuda')
        self.model = WorldModel(cfg).to(self.device)
        self.teacher = teacher
        
        self.optim = torch.optim.Adam([
            {'params': self.model._encoder.parameters(), 'lr': self.cfg.lr * self.cfg.enc_lr_scale},
            {'params': self.model._dynamics.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._Qs.parameters()},
            {'params': self.model._task_emb.parameters() if self.cfg.multitask else []}
        ], lr=self.cfg.lr)
        self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.model.eval()

    def load(self, fp, strict=False):
        state_dict = torch.load(fp, weights_only=False)
        return self.model.load_state_dict(state_dict["model"], strict=strict)

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        obs = obs.to(self.device, non_blocking=True)
        z = self.model.encode(obs, task) 
        a = self.model.pi(z, task)[int(not eval_mode)] 
        return a.cpu()

    def update(self, buffer):
        obs, action, reward, task = buffer.sample()
        with torch.no_grad():
            next_z = self.model.encode(obs[1:], task)
            
            teacher_reward = None
            if self.teacher is not None:
                # TD-MPC-OPT: Reward Distillation
                # Slice ManiSkill3 data to match Teacher's MT30 expectations
                t_obs = obs[0][:, :24] 
                t_action = action[0][:, :6]
                
                # FIX: Provide a default task tensor if the environment is single-task
                if task is None:
                    t_task = torch.zeros(t_obs.shape[0], dtype=torch.long, device=self.device)
                else:
                    t_task = torch.zeros_like(task)
                    
                teacher_z = self.teacher.model.encode(t_obs, t_task)
                teacher_reward = self.teacher.model.reward(teacher_z, t_action, t_task)

        self.optim.zero_grad(set_to_none=True)
        self.model.train()

        # Consistency rollout
        z = self.model.encode(obs[0], task)
        consistency_loss = 0
        for t in range(self.cfg.horizon):
            z = self.model.next(z, action[t], task)
            consistency_loss += F.mse_loss(z, next_z[t]) * self.cfg.rho**t

        # Distillation loss: MSE between teacher and student rewards
        distill_loss = torch.tensor(0.0, device=self.device)
        if teacher_reward is not None:
            student_reward = self.model.reward(self.model.encode(obs[0], task), action[0], task)
            distill_loss = F.mse_loss(student_reward, teacher_reward)

        # Total combined loss
        d_coef = getattr(self.cfg, 'distill_coef', 0.4)
        total_loss = (self.cfg.consistency_coef * (consistency_loss / self.cfg.horizon) + 
                      d_coef * distill_loss)

        total_loss.backward()
        self.optim.step()
        self.model.eval()
        return {"total_loss": total_loss.item(), "distill_loss": distill_loss.item()}