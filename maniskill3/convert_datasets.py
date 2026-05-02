import os
from pathlib import Path

import h5py
import numpy as np
import torch

TASKS = [
    "PushCube-v1",
    "PickCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1"
]

DEMO_DIR = Path(os.path.expanduser("~/.maniskill/demos/"))
OUTPUT_DIR = Path("/root/td-mpc-opt/maniskill3/offline_data/")


def get_raw_observations(traj_obs) -> np.ndarray:
    if isinstance(traj_obs, h5py.Group):
        if "state" in traj_obs:
            return np.array(traj_obs["state"])
        first_key = list(traj_obs.keys())[0]
        return np.array(traj_obs[first_key])
    return np.array(traj_obs)


def process_episode(traj, ep_idx: int, out_dir: Path) -> int:
    raw_obs = get_raw_observations(traj["obs"])
    raw_actions = np.array(traj["actions"])

    if "rewards" in traj:
        raw_rewards = np.array(traj["rewards"])
    else:
        raw_rewards = np.zeros((raw_actions.shape[0], 1))

    obs_sliced = raw_obs[:, :24] if raw_obs.shape[-1] >= 24 else raw_obs
    act_sliced = (
        raw_actions[:, :6] if raw_actions.shape[-1] >= 6 else raw_actions
    )

    torch.save({
        "obs": torch.tensor(obs_sliced, dtype=torch.float32),
        "action": torch.tensor(act_sliced, dtype=torch.float32),
        "reward": torch.tensor(raw_rewards, dtype=torch.float32)
    }, out_dir / f"ep_{ep_idx}.pt")

    return len(act_sliced)


def process_task(task_name: str):
    task_dir = DEMO_DIR / task_name
    try:
        h5_path = next(task_dir.rglob("*.state.*.h5"))
    except StopIteration:
        print(f"[ERROR] State file missing for {task_name}.")
        return

    print(f"Processing {task_name} from {h5_path.name}...")
    task_out_dir = OUTPUT_DIR / task_name
    task_out_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    with h5py.File(h5_path, "r") as f:
        trajs = [k for k in f.keys() if k.startswith("traj_")]

        for ep_idx, traj_key in enumerate(trajs):
            total_steps += process_episode(
                f[traj_key], ep_idx, task_out_dir
            )

    print(f"[{task_name}] Saved {len(trajs)} episodes "
          f"({total_steps} steps) to {task_out_dir}\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Starting Final Data Conversion...\n")

    for task in TASKS:
        process_task(task)

    print("Conversion Complete! Matrices are fully loaded.")


if __name__ == "__main__":
    main()