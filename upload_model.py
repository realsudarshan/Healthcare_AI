from huggingface_hub import HfApi

api = HfApi()
repo_id = "realsudarshan/chest-xray-classifier"

print(f"Uploading checkpoint to Model Repository: {repo_id}")

try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj="checkpoints/checkpoint_epoch_10.pt",
        path_in_repo="checkpoint_epoch_10.pt",
        repo_id=repo_id,
        repo_type="model"
    )
    print("Upload complete!")
except Exception as e:
    print(f"Upload failed: {e}")
