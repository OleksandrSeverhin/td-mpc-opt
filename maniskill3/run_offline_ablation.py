import os
import subprocess
import time
from datetime import datetime

# The 5 mathematically sound datasets
TASKS = [
    "PushCube-v1",
    "PickCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1"
]

# The 3 decay schedules
SCHEDULES = ["constant", "linear", "cosine"]

DATA_BASE_DIR = "/root/td-mpc-opt/maniskill3/offline_data"

def main():
    print("="*60)
    print("  STARTING MASSIVE OFFLINE ABLATION STUDY")
    print("="*60)

    for task in TASKS:
        for schedule in SCHEDULES:
            dataset_path = os.path.join(DATA_BASE_DIR, task)
            exp_name = f"offline_{schedule}_{task}"
            
            print("-" * 60)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training: {task} | Schedule: {schedule}")
            print(f"Dataset: {dataset_path}")
            print("-" * 60)

            # Construct the command list exactly as we did in Bash
            cmd = [
                "python", "train_offline_distill.py",
                f"env_id={task}",
                f"+dataset_path={dataset_path}",
                f"exp_name={exp_name}",
                f"+distillation.schedule={schedule}",
                "+distillation.d_coef=0.4",
                "steps=1000000",
                "batch_size=512"
            ]

            # Execute the command
            try:
                # check=True ensures that if the training script crashes, it raises an exception here
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n[FATAL ERROR] {exp_name} crashed with exit code {e.returncode}.")
                print("Halting the entire ablation study to prevent cascading failures.")
                return

            print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished {exp_name}. Cooling down GPU for 5 seconds...\n")
            time.sleep(5)

    print("="*60)
    print("  ALL EXPERIMENTS COMPLETE. Check the 'logs/' folder!")
    print("="*60)

if __name__ == "__main__":
    main()