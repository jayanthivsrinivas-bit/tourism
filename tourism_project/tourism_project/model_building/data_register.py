from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "Fitjv/tourism-dataset"   # dataset repo name
repo_type = "dataset"

# Init API client with token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment. Add it via Colab secrets.")
api = HfApi(token=HF_TOKEN)

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset repo '{repo_id}' not found. Creating new one...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset repo '{repo_id}' created.")

# Upload local dataset folder
BASE = "Project/tourism_project"
LOCAL_DATA_FOLDER = os.path.join(BASE, "data")

api.upload_folder(
    folder_path=LOCAL_DATA_FOLDER,
    repo_id=repo_id,
    repo_type=repo_type,
)
print(f"✅ Data uploaded successfully to {repo_id}")
