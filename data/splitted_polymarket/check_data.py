from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./swm-bench",
    repo_id="ulab-ai/swm-bench",
    repo_type="dataset",
)
