import os
import h5py
import torch
import numpy as np
from pathlib import Path

TASKS = [
    "PushCube-v1",
    "PickCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1"
]

DEMO_DIR = Path(os.path.expanduser("~/.maniskill/demos/"))
OUTPUT_DIR = Path("/root/td-mpc-opt/maniskill3/offline_data/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_task(task_name):
    task_dir = DEMO_DIR / task_name
    h5_path = None
    
    # 1. Target the newly unpacked state files
    for path in task_dir.rglob("*.state.*.h5"):
        h5_path = path
        break
        
    if h5_path is None:
        print(f"[ERROR] Unpacked state file missing for {task_name}.")
        return

    print(f"Processing {task_name} from {h5_path.name}...")
    task_out_dir = OUTPUT_DIR / task_name
    os.makedirs(task_out_dir, exist_ok=True)

    total_transitions = 0
    
    with h5py.File(h5_path, 'r') as f:
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]
        
        for ep_idx, traj_key in enumerate(traj_keys):
            traj = f[traj_key]
            
            # 2. BULLETPROOF OBS EXTRACTION
            if isinstance(traj['obs'], h5py.Group):
                if 'state' in traj['obs']:
                    raw_obs = np.array(traj['obs']['state'])
                else:
                    key = list(traj['obs'].keys())[0]
                    raw_obs = np.array(traj['obs'][key])
            else:
                raw_obs = np.array(traj['obs'])
                
            raw_actions = np.array(traj['actions'])
            
            if 'rewards' in traj:
                raw_rewards = np.array(traj['rewards'])
            else:
                raw_rewards = np.zeros((raw_actions.shape[0], 1))

            # 3. Apply the 24-dimensional slice
            obs_sliced = raw_obs[:, :24] if raw_obs.shape[-1] >= 24 else raw_obs
            actions_sliced = raw_actions[:, :6] if raw_actions.shape[-1] >= 6 else raw_actions
            
            obs_tensor = torch.tensor(obs_sliced, dtype=torch.float32)
            action_tensor = torch.tensor(actions_sliced, dtype=torch.float32)
            reward_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
            
            torch.save({
                'obs': obs_tensor, 
                'action': action_tensor, 
                'reward': reward_tensor
            }, task_out_dir / f"ep_{ep_idx}.pt")
            
            total_transitions += len(actions_sliced)

    print(f"[{task_name}] Saved {len(traj_keys)} episodes ({total_transitions} steps) to {task_out_dir}\n")

def main():
    print("Starting Final Data Conversion...\n")
    for task in TASKS:
        process_task(task)
    print("Conversion Complete! Matrices are fully loaded.")

if __name__ == "__main__":
    main()