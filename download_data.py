from huggingface_hub import hf_hub_download
import os

# Target folder for dataset chunks
target_dir = "data/"
os.makedirs(target_dir, exist_ok=True)

# Chunk filenames: chunk_0.pt to chunk_3.pt
for i in range(4):  # 0 to 3 inclusive
    filename = f"mt30/chunk_{i}.pt"
    
    local_path = hf_hub_download(
        repo_id="nicklashansen/tdmpc2",
        repo_type="dataset",
        filename=filename,
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print(f"Downloaded {filename} to {local_path}")

# Download the model to tdmpc2/models
model_dir = "tdmpc2/models"
os.makedirs(model_dir, exist_ok=True)

model_path = hf_hub_download(
    repo_id="nicklashansen/tdmpc2",
    repo_type="model",
    filename="multitask/mt30-317M.pt",
    local_dir=model_dir,
    local_dir_use_symlinks=False
)

print(f"Model downloaded to {model_path}")