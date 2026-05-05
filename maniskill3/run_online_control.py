import os
import subprocess
import time
from datetime import datetime
from huggingface_hub import HfApi

# The Control Variables
TASK = "PushCube-v1"
STEPS = 1000000
LOG_DIR = "/root/td-mpc-opt/maniskill3/logs"
REPO_ID = "oleksandr-severhin/td-mpc-opt"

api = HfApi(token=os.getenv("HF_TOKEN"))

def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def main():
    exp_name = f"online_control_{TASK}_1M"

    print("=" * 60)
    print(f"[{get_timestamp()}] STARTING ONLINE CONTROL EXPERIMENT")
    print(f"Task: {TASK} | Steps: {STEPS}")
    print("=" * 60)

    # Note: We are using train.py (Online RL), NOT train_offline_distill.py
    cmd = [
        "python", "train.py",
        f"env_id={TASK}",
        f"exp_name={exp_name}",
        f"steps={STEPS}"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n[FATAL] {exp_name} crashed (exit code {e.returncode}).")
        return

    print(f"[{get_timestamp()}] Finished {exp_name}. Uploading to HuggingFace...\n")
    
    try:
        api.upload_folder(
            folder_path=LOG_DIR,
            repo_id=REPO_ID,
            repo_type="model",
            allow_patterns=f"**/{exp_name}**/model.pt",
            commit_message=f"Auto-backup online control model: {exp_name}"
        )
        print(f"[{get_timestamp()}] Upload successful.\n")
    except Exception as e:
        print(f"\n[WARNING] HuggingFace upload failed: {e}")

if __name__ == "__main__":
    main()