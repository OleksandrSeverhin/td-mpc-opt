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

# 2. Define checkpoints and directories
TEACHER_CKPT = "/root/td-mpc-opt/tdmpc2/models/multitask/mt30-317M.pt"
RESULTS_DIR = "results"

# Create the results folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Batch Distillation Pipeline...")
    print(f"Logs will be saved to the '{RESULTS_DIR}/' directory.\n")

    for task in TASKS:
        print("="*60)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Launching: {task}")
        print("="*60)
        
        # Define the exact command and arguments
        cmd = [
            "python", "train_distill.py",
            f"env_id={task}",
            "model_size=1",
            f"checkpoint={TEACHER_CKPT}",
            f"exp_name=ms3_distill_1M_{task}",
            "steps=1000000"
        ]
        
        # Define the output file path for this specific task
        log_file_path = os.path.join(RESULTS_DIR, f"{task}_output.txt")
        
        # Open the file and start the process
        with open(log_file_path, "w") as log_file:
            # We use Popen to capture the output in real-time
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, # Redirect errors to the same output stream
                text=True,
                bufsize=1 # Line buffered
            )
            
            # Read the output line-by-line as the training script runs
            for line in process.stdout:
                print(line, end="")  # Print to your terminal screen
                log_file.write(line) # Save to the text file
                log_file.flush()     # Force write to disk immediately (safe against crashes)
            
            # Wait for the process to officially finish
            process.wait()
            
            if process.returncode != 0:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] WARNING: {task} stopped with error code {process.returncode}")
            else:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Successfully finished {task}")
        
        # Sleep for 10 seconds to allow PyTorch to fully release GPU VRAM
        print("Flushing GPU Memory. Waiting 10 seconds...\n")
        time.sleep(10)

    print("========================================================")
    print("ALL TASKS COMPLETED. Check the 'results/' folder for logs.")
    print("========================================================")

if __name__ == "__main__":
    main()