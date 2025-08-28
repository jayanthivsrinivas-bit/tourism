from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "Fitjv/tourism-dataset"   # ✅ dataset repo name
repo_type = "dataset"

# Init API client with token
api = HfApi(token=os.getenv("HF_TOKEN"))

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new one...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Upload local dataset folder
api.upload_folder(
    folder_path="Project/tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"✅ Data uploaded successfully to {repo_id}")
