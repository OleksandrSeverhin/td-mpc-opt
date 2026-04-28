import os
import torch
import numpy as np
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import wandb

from common.parser import parse_cfg
from common.seed import set_seed
from tdmpc2 import TDMPC2

class OfflineBuffer:
    def __init__(self, data_dir, cfg):
        self.cfg = cfg
        self.episodes = []
        
        print(f"Loading offline dataset from: {data_dir}...")
        skipped = 0
        for fn in os.listdir(data_dir):
            if fn.endswith('.pt'):
                ep = torch.load(os.path.join(data_dir, fn))
                
                # Validation check: Only load fully formed 24-dimension tensors
                if ep['obs'].shape[-1] == 24:
                    self.episodes.append(ep)
                else:
                    skipped += 1
                    
        self.num_eps = len(self.episodes)
        print(f"Successfully loaded {self.num_eps} expert episodes into RAM.")
        if skipped > 0:
            print(f"Ignored {skipped} empty or corrupted episodes.")

    def sample(self):
        batch_obs, batch_act, batch_rew = [], [], []
        
        for _ in range(self.cfg.batch_size):
            ep = self.episodes[np.random.randint(self.num_eps)]
            L = ep['obs'].shape[0]
            
            if L <= 6:
                start = 0 
            else:
                start = np.random.randint(0, L - 6)
                
            obs = ep['obs'][start:start+6]
            act = ep['action'][start:start+5]
            rew = ep['reward'][start:start+5]
            
            if obs.shape[0] < 6:
                obs = torch.cat([obs, obs[-1:].repeat(6 - obs.shape[0], 1)])
            if act.shape[0] < 5:
                act = torch.cat([act, act[-1:].repeat(5 - act.shape[0], 1)])
            if rew.shape[0] < 5:
                rew = torch.cat([rew, rew[-1:].repeat(5 - rew.shape[0], 1)])
                
            batch_obs.append(obs)
            batch_act.append(act)
            batch_rew.append(rew)
            
        obs_out = torch.stack(batch_obs, dim=1).to('cuda')
        act_out = torch.stack(batch_act, dim=1).to('cuda')
        rew_out = torch.stack(batch_rew, dim=1).to('cuda')
        
        task_out = torch.zeros(self.cfg.batch_size, dtype=torch.long, device='cuda')
        
        return obs_out, act_out, rew_out, task_out


@hydra.main(config_name='config', config_path='.', version_base=None)
def train_offline(cfg: dict):
    assert torch.cuda.is_available()
    
    OmegaConf.set_struct(cfg, False)
    cfg.episode_length = 100
    cfg.horizon = 5
    
    cfg = parse_cfg(cfg)
    
    cfg.action_dim = 6
    cfg.obs_shape = {'state': [24]}
    
    set_seed(cfg.seed)
    
    wandb.init(
        project="tdmpc2_offline_distillation", 
        name=cfg.exp_name, 
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    dataset_path = getattr(cfg, 'dataset_path', None)
    if dataset_path is None:
        raise ValueError("You must provide a dataset_path")
    buffer = OfflineBuffer(dataset_path, cfg)
    
    print("\nLoading 317M Teacher Model...")
    teacher_cfg = cfg.copy()
    teacher_cfg.model_size = 317
    teacher_cfg.multitask = True 
    teacher_cfg.tasks = [str(i) for i in range(30)]
    teacher_cfg.action_dims = [6] * 30
    teacher_cfg.episode_lengths = [cfg.episode_length] * 30
    
    teacher_cfg.num_enc_layers = 5
    teacher_cfg.enc_dim = 4096
    teacher_cfg.mlp_dim = 4096
    teacher_cfg.latent_dim = 1376
    teacher_cfg.num_q = 8
    teacher_cfg.task_dim = 96
    
    teacher_agent = TDMPC2(teacher_cfg)
    teacher_checkpoint = "/root/td-mpc-opt/tdmpc2/models/multitask/mt30-317M.pt"
    
    teacher_agent.load(teacher_checkpoint)
    teacher_agent.model.eval()
    for param in teacher_agent.model.parameters():
        param.requires_grad = False
        
    print("Initializing 1M Student Model...")
    student_cfg = cfg.copy()
    student_cfg.model_size = 1
    student_cfg.num_enc_layers = 2
    student_cfg.enc_dim = 256
    student_cfg.mlp_dim = 512
    student_cfg.latent_dim = 512
    student_agent = TDMPC2(student_cfg, teacher_model=teacher_agent)

    student_agent._get_horizon = lambda *args, **kwargs: 5

    print(f"\nStarting Offline Distillation ({cfg.distillation.schedule} decay)...")
    for step in range(cfg.steps):
        
        stats = student_agent.update(buffer, step=step)
        
        if step % 5000 == 0:
            print(f"Step: {step}/{cfg.steps} | Total Loss: {stats['total_loss']:.4f} | Distill Coef: {stats['distillation_coef']:.4f}")
            
        wandb.log(stats, step=step)

    save_dir = Path(f"logs/{cfg.exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "model.pt"
    student_agent.save(str(save_path))
    
    print(f"\nOffline Training Complete! Model saved to {save_path}")
    wandb.finish()

if __name__ == "__main__":
    train_offline()