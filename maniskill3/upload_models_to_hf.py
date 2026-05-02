from huggingface_hub import HfApi

HF_TOKEN = ""
REPO_ID = "oleksandr-severhin/td-mpc-opt"
LOG_DIR = "/root/td-mpc-opt/maniskill3/logs"


def main():
    api = HfApi(token=HF_TOKEN)
    print(f"Connecting to Hugging Face as {REPO_ID}...")

    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    print("Repository confirmed. Scanning directory...")

    msg = "Upload final 100k step Student models for ablation study"
    api.upload_folder(
        folder_path=LOG_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        allow_patterns="**/model.pt",
        commit_message=msg
    )

    print("Upload complete! All final models are safely backed up.")


if __name__ == "__main__":
    main()