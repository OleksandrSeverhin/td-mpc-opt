import os
import subprocess
import time
from datetime import datetime

# 1. Define your chosen ManiSkill3 tasks
TASKS = [
    "PushCube-v1",
    "PickCube-v1",
    "StackCube-v1",
    "PegInsertionSide-v1",
    "PushT-v1",
    "PlugCharger-v1",
    "TurnFaucet-v1",
    "OpenCabinetDrawer-v1"
]

# 2. Define checkpoint and directories
TEACHER_CKPT = "/root/td-mpc-opt/tdmpc2/models/multitask/mt30-317M.pt"
RESULTS_DIR = "results"

# Create the results folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Teacher Evaluation Pipeline...")
    print(f"Logs will be saved to the '{RESULTS_DIR}/' directory.\n")

    for task in TASKS:
        print("="*60)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Evaluating Teacher on: {task}")
        print("="*60)
        
        # Define the exact command for Baseline Evaluation
        cmd = [
            "python", "eval_teacher.py",
            f"env_id={task}",
            f"checkpoint={TEACHER_CKPT}",
            "render=false"
        ]
        
        # Save output to a distinct file with "_eval_output"
        log_file_path = os.path.join(RESULTS_DIR, f"{task}_eval_output.txt")
        
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
    print("ALL EVALUATIONS COMPLETED. Check the 'results/' folder for scores.")
    print("========================================================")

if __name__ == "__main__":
    main()