import argparse
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


def get_time() -> str:
    return datetime.now().strftime("%H:%M:%S")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Student models.")
    parser.add_argument(
        "--schedule",
        type=str,
        required=True,
        choices=["constant", "linear", "cosine"],
        help="Distillation coefficient schedule."
    )
    return parser.parse_args()


def evaluate_task(task: str, schedule: str, results_dir: str) -> None:
    print("-" * 60)
    print(f"[{get_time()}] Evaluating Student on: {task}")
    print("-" * 60)

    checkpoint_path = (
        f"/root/td-mpc-opt/maniskill3/logs/"
        f"ms3_ablation_{schedule}_{task}/model.pt"
    )

    cmd = [
        "python", "eval_student.py",
        f"env_id={task}",
        f"checkpoint={checkpoint_path}",
        "render=false"
    ]

    log_path = os.path.join(results_dir, f"{task}_eval_output.txt")

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
                f"\n[{get_time()}] WARNING: {task} stopped "
                f"with error code {process.returncode}"
            )
        else:
            print(f"\n[{get_time()}] Successfully evaluated {task}")


def main() -> None:
    args = parse_arguments()
    schedule = args.schedule
    results_dir = f"results_for_student_{schedule}"
    
    os.makedirs(results_dir, exist_ok=True)

    print(f"[{get_time()}] Starting Student Evaluation Pipeline...")
    print(f"Testing models trained with '{schedule}' decay.")
    print(f"Logs will be saved to: {results_dir}/\n")

    for task in TASKS:
        evaluate_task(task, schedule, results_dir)
        time.sleep(5)

    print("-" * 60)
    print(f"ALL EVALUATIONS COMPLETED FOR {schedule.upper()} DECAY.")
    print(f"Check the '{results_dir}/' folder for scores.")
    print("-" * 60)


if __name__ == "__main__":
    main()