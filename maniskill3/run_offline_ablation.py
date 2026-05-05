import os
import subprocess
import time
from datetime import datetime

TASKS = [
    "PushCube-v1",
    #"PickCube-v1",
    #"StackCube-v1",
    #"PegInsertionSide-v1",
    #"PlugCharger-v1"
]

SCHEDULES = ["constant", "linear", "cosine"]
DATA_BASE_DIR = "/root/td-mpc-opt/maniskill3/offline_data"


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_experiment(task: str, schedule: str) -> bool:
    dataset_path = os.path.join(DATA_BASE_DIR, task)
    exp_name = f"offline_{schedule}_{task}_1M"

    print("-" * 60)
    print(f"[{get_timestamp()}] Training: {task} | Schedule: {schedule}")
    print(f"Dataset: {dataset_path}")
    print("-" * 60)

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

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] {exp_name} crashed (exit code {e.returncode}).")
        print("Halting ablation study to prevent cascading failures.")
        return False

    print(f"[{get_timestamp()}] Finished {exp_name}. Cooling GPU...\n")
    time.sleep(5)
    return True


def main():
    print("=" * 60)
    print(" STARTING OFFLINE ABLATION STUDY")
    print("=" * 60)

    for task in TASKS:
        for schedule in SCHEDULES:
            success = run_experiment(task, schedule)
            if not success:
                return

    print("=" * 60)
    print(" ALL EXPERIMENTS COMPLETE. Check the 'logs/' folder!")
    print("=" * 60)


if __name__ == "__main__":
    main()