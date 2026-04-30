from huggingface_hub import HfApi
import os

# Initialize the API using your token
HF_TOKEN=""

api = HfApi(token=HF_TOKEN)

# Update these variables to match your setup
REPO_ID = "oleksandr-severhin/td-mpc-opt"
LOG_DIR = "/root/td-mpc-opt/maniskill3/logs"

print(f"Connecting to Hugging Face as {REPO_ID}...")

# 2. THE FIX: Force create the repository if it doesn't exist
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
print("Repository confirmed. Scanning directory...")

# 3. Upload only the final models
api.upload_folder(
    folder_path=LOG_DIR,
    repo_id=REPO_ID,
    repo_type="model",
    allow_patterns="**/model.pt",
    commit_message="Upload final 100k step Student models for ManiSkill3 ablation study"
)

print("Upload complete! All final models are safely backed up.")