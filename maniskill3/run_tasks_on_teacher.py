import os
import subprocess
import time
from datetime import datetime

TASKS = [
    "PushCube-v1",
    "PickCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
    "PlugCharger-v1",
    "TurnFaucet-v1",
    "OpenCabinetDrawer-v1",
    "PushT-v1"
]

TEACHER_CKPT = "/root/td-mpc-opt/tdmpc2/models/multitask/mt30-317M.pt"
RESULTS_DIR = "results"


def get_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def evaluate_task(task: str):
    print("-" * 60)
    print(f"[{get_time()}] Evaluating Teacher on: {task}")
    print("-" * 60)

    cmd = [
        "python", "eval_teacher.py",
        f"env_id={task}",
        f"checkpoint={TEACHER_CKPT}",
        "render=false"
    ]

    log_path = os.path.join(RESULTS_DIR, f"{task}_eval_output.txt")

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        process.wait()

        if process.returncode != 0:
            print(
                f"\n[{get_time()}] WARNING: {task} "
                f"stopped with error code {process.returncode}"
            )
        else:
            print(f"\n[{get_time()}] Successfully evaluated {task}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[{get_time()}] Starting Teacher Evaluation Pipeline...")
    print(f"Logs will be saved to '{RESULTS_DIR}/'.\n")

    for task in TASKS:
        evaluate_task(task)
        time.sleep(5)

    print("-" * 60)
    print(f"ALL EVALUATIONS COMPLETED. Check '{RESULTS_DIR}/'.")
    print("-" * 60)


if __name__ == "__main__":
    main()