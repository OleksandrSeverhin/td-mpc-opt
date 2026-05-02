from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "nicklashansen/tdmpc2"
DATA_DIR = Path("data/")
MODEL_DIR = Path("tdmpc2/models/")


def download_dataset_chunks(num_chunks: int = 4):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(num_chunks):
        filename = f"mt30/chunk_{i}.pt"
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=str(DATA_DIR),
            local_dir_use_symlinks=False
        )
        print(f"Downloaded {filename} to {local_path}")


def download_model(filename: str):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="model",
        filename=filename,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to {local_path}")


def main():
    download_dataset_chunks(num_chunks=4)
    download_model("multitask/mt30-317M.pt")


if __name__ == "__main__":
    main()