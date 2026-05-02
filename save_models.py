import os

from huggingface_hub import HfApi

FOLDER_PATH = (
    "/root/td-mpc-opt/logs/mt30/1/"
    "mt30_1M_steps_50_four_phase/models"
)
REPO_ID = "oleksandr-severhin/td-mpc-opt"


def main() -> None:
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable is missing.")

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=FOLDER_PATH,
        repo_id=REPO_ID,
        repo_type="model",
    )


if __name__ == "__main__":
    main()