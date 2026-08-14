import os
from huggingface_hub import HfApi

api = HfApi()
repo_id = "realsudarshan/chest-xray-demo"

print(f"Creating Space Repository: {repo_id}")
try:
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
except Exception as e:
    print(f"Error or already exists: {e}")

print("Uploading files...")
api.upload_folder(
    folder_path="hf_deployment",
    repo_id=repo_id,
    repo_type="space"
)
print("Deployment complete! Live at: https://huggingface.co/spaces/" + repo_id)
