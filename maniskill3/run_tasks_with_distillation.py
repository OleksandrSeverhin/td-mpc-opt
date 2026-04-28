import os
import subprocess
import time
import argparse
from datetime import datetime

# 1. Define your chosen ManiSkill3 tasks
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

def main():
    # Set up argument parsing so you can choose the schedule from the terminal
    parser = argparse.ArgumentParser(description="Automate Student Evaluation across tasks.")
    parser.add_argument(
        '--schedule', 
        type=str, 
        required=True, 
        choices=['constant', 'linear', 'cosine'],
        help="The distillation coefficient schedule used during training."
    )
    args = parser.parse_args()
    
    schedule = args.schedule
    
    # 2. Set up the dynamic results directory
    results_dir = f"results_for_student_{schedule}"
    os.makedirs(results_dir, exist_ok=True)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Student Evaluation Pipeline...")
    print(f"Testing models trained with '{schedule}' decay.")
    print(f"Logs will be saved to: {results_dir}/\n")

    for task in TASKS:
        print("="*60)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluating Student on: {task}")
        print("="*60)
        
        # 3. Define the exact checkpoint path based on the schedule and task
        # Note: Adjust this path if your Hydra logs are named differently!
        checkpoint_path = f"/root/td-mpc-opt/maniskill3/logs/ms3_ablation_{schedule}_{task}/model.pt"
        
        # Define the exact command using the eval_student.py script
        cmd = [
            "python", "eval_student.py",
            f"env_id={task}",
            f"checkpoint={checkpoint_path}",
            "render=false"
        ]
        
        # Save output to a distinct file in the dynamically created folder
        log_file_path = os.path.join(results_dir, f"{task}_eval_output.txt")
        
        with open(log_file_path, "w") as log_file:
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
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] WARNING: {task} stopped with error code {process.returncode}")
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Successfully finished evaluating {task}")
        
        # Give the GPU a moment to clear VRAM before loading the next environment
        time.sleep(5)

    print("========================================================")
    print(f"ALL EVALUATIONS COMPLETED FOR {schedule.upper()} DECAY.")
    print(f"Check the '{results_dir}/' folder for scores.")
    print("========================================================")

if __name__ == "__main__":
    main()