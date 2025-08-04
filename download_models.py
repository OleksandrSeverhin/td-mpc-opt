from huggingface_hub import HfApi, hf_hub_download
import os

# Load your token from environment variable
hf_token = os.getenv("HF_TOKEN")
assert hf_token is not None, "HF_TOKEN environment variable is not set."

models = ['mt30_1M_steps_50_const_final.pt', 'mt30_1M_steps_50_cosine_decay_final.pt', 
          'mt30_1M_steps_50_decrease_final.pt', 'mt30_1M_steps_50_four_phase_final.pt', 
          'mt30_1M_steps_50_increase_final.pt', ' mt30_1M_steps_50_linear_decay_final.pt']

repo_id = "oleksandr-severhin/td-mpc-opt"

for model in models:
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=model,
        token=hf_token,
        cache_dir="/root/td-mpc-opt/tdmpc2/models" 
    )
    print(f"Model downloaded to: {local_path}")