import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
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
    """
    100% VRAM-Resident Vectorized Buffer.
    Bypasses the CPU completely to feed the Tensor Cores.
    """
    def __init__(self, data_dir, cfg):
        self.cfg = cfg
        print(f"Loading offline dataset from: {data_dir}...")
        
        all_obs, all_act, all_rew = [], [], []
        skipped = 0
        
        for fn in os.listdir(data_dir):
            if fn.endswith('.pt'):
                ep = torch.load(os.path.join(data_dir, fn))
                
                # Filter out corrupted 0-dim files from previous runs
                if ep['obs'].shape[-1] != 24:
                    skipped += 1
                    continue
                    
                obs = ep['obs']
                act = ep['action']
                rew = ep['reward']
                L = obs.shape[0]
                
                # Edge-case padding for very short episodes
                if L < 6:
                    obs = torch.cat([obs, obs[-1:].repeat(6 - L, 1)])
                if act.shape[0] < 5:
                    act = torch.cat([act, act[-1:].repeat(5 - act.shape[0], 1)])
                if rew.shape[0] < 5:
                    rew = torch.cat([rew, rew[-1:].repeat(5 - rew.shape[0], 1)])
                    
                # Pre-slice all possible 6-step sequences to eliminate Python loops
                L_pad = obs.shape[0]
                for i in range(L_pad - 5):
                    all_obs.append(obs[i:i+6])
                    all_act.append(act[i:i+5])
                    all_rew.append(rew[i:i+5])

        print("Pushing entire dataset directly to GPU VRAM...")
        # Stack into massive tensors and push to CUDA exactly once
        self.obs_data = torch.stack(all_obs).to('cuda')
        self.act_data = torch.stack(all_act).to('cuda')
        self.rew_data = torch.stack(all_rew).to('cuda')
        self.N = self.obs_data.shape[0]
        
        # Pre-allocate task tensor to prevent memory fragmenting every step
        self.task_out = torch.zeros(self.cfg.batch_size, dtype=torch.long, device='cuda')
        
        print(f"Successfully loaded {self.N} sequences into VRAM.")
        if skipped > 0:
            print(f"Ignored {skipped} empty or corrupted episodes.")

    def sample(self):
        # Pure GPU vectorized sampling
        idxs = torch.randint(0, self.N, (self.cfg.batch_size,), device='cuda')
        
        # Use .contiguous() to prevent invisible CPU memory syncs
        obs_out = self.obs_data[idxs].transpose(0, 1).contiguous()
        act_out = self.act_data[idxs].transpose(0, 1).contiguous()
        rew_out = self.rew_data[idxs].transpose(0, 1).contiguous()
        
        return obs_out, act_out, rew_out, self.task_out


@hydra.main(config_name='config', config_path='.', version_base=None)
def train_offline(cfg: dict):
    assert torch.cuda.is_available()
    
    # --- HARDWARE UNLOCKS ---
    # Unleash the RTX 3090 Tensor Cores
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # --- HYDRA UNLOCK & BYPASSES ---
    OmegaConf.set_struct(cfg, False)
    cfg.episode_length = 100
    cfg.horizon = 5 # Standard TD-MPC2 horizon
    
    cfg = parse_cfg(cfg)
    
    # Explicitly enforce dimensions for the 24-dim state vectors
    cfg.action_dim = 6
    cfg.obs_shape = {'state': [24]}
    
    set_seed(cfg.seed)
    
    # 1. Setup Tracking
    wandb.init(
        project="tdmpc2_offline_distillation", 
        name=cfg.exp_name, 
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # 2. Initialize Buffer
    dataset_path = getattr(cfg, 'dataset_path', None)
    if dataset_path is None:
        raise ValueError("You must provide a dataset_path")
    buffer = OfflineBuffer(dataset_path, cfg)
    
    # 3. Initialize Teacher (317M, Frozen)
    print("\nLoading 317M Teacher Model...")
    teacher_cfg = cfg.copy()
    teacher_cfg.model_size = 317
    teacher_cfg.multitask = True 
    teacher_cfg.tasks = [str(i) for i in range(30)]
    teacher_cfg.action_dims = [6] * 30
    teacher_cfg.episode_lengths = [cfg.episode_length] * 30
    
    # Architecture params discovered from MT30-317M checkpoint
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
        
    # 4. Initialize Student (1M, Active)
    print("Initializing 1M Student Model...")
    student_cfg = cfg.copy()
    student_cfg.model_size = 1
    student_cfg.num_enc_layers = 2
    student_cfg.enc_dim = 256
    student_cfg.mlp_dim = 512
    student_cfg.latent_dim = 512
    student_agent = TDMPC2(student_cfg, teacher_model=teacher_agent)

    # CRITICAL: Bypass dynamic horizon curriculum
    student_agent._get_horizon = lambda *args, **kwargs: 5

    # 5. OFFLINE TRAINING LOOP
    print(f"\nStarting Offline Distillation ({cfg.distillation.schedule} decay)...")
    for step in range(cfg.steps):
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            stats = student_agent.update(buffer, step=step)
        
        if step % 5000 == 0:
            print(f"Step: {step}/{cfg.steps} | Total Loss: {stats['total_loss']:.4f} | Distill Coef: {stats['distillation_coef']:.4f}")
            
        if step > 0 and step % 10000 == 0:
            wandb.log(stats, step=step)

        # Intermediate checkpoint for thesis safety
        if step > 0 and step % 50000 == 0:
            save_dir = Path(f"logs/{cfg.exp_name}")
            save_dir.mkdir(parents=True, exist_ok=True)
            intermediate_path = save_dir / f"model_{step}.pt"
            student_agent.save(str(intermediate_path))

    # 6. Final Save
    save_dir = Path(f"logs/{cfg.exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "model.pt"
    student_agent.save(str(save_path))
    
    print(f"\nOffline Training Complete! Final model saved to {save_path}")
    wandb.finish()

if __name__ == "__main__":
    train_offline()