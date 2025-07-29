from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/root/td-mpc-opt/logs/mt30/1/mt30_1M_steps_50_four_phase/models",
    repo_id="oleksandr-severhin/td-mpc-opt",
    repo_type="model",
)