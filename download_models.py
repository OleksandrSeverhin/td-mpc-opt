import os
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "oleksandr-severhin/td-mpc-opt"
CACHE_DIR = "/root/td-mpc-opt/tdmpc2/models"

MODELS = [
    "mt30_1M_steps_50_const_final.pt",
    "mt30_1M_steps_50_cosine_decay_final.pt",
    "mt30_1M_steps_50_decrease_final.pt",
    "mt30_1M_steps_50_four_phase_final.pt",
    "mt30_1M_steps_50_increase_final.pt",
    "mt30_1M_steps_50_linear_decay_final.pt"
]


def download_models(token: str) -> None:
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=model,
            token=token,
            cache_dir=CACHE_DIR
        )
        print(f"Downloaded to: {local_path}")


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is missing.")

    download_models(token)


if __name__ == "__main__":
    main()