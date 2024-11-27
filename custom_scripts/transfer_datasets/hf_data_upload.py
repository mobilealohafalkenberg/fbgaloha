import os
from huggingface_hub import HfApi, create_repo, upload_folder
import sys
from pathlib import Path
from typing import Optional, List

def upload_dataset_to_hf(
    dataset_path: str,
    repo_name: str,
    repo_type: str = "dataset",
    ignore_patterns: Optional[List[str]] = None
):
    """
    Upload a dataset to HuggingFace Hub using cached credentials

    Args:
        dataset_path: Local path to the dataset files
        repo_name: Name for the HuggingFace repository
        repo_type: Type of repository (default: "dataset")
        ignore_patterns: List of file patterns to ignore during upload
    """
    try:
        print("\n🤗 Starting HuggingFace Dataset Upload Process...")
        
        # Validate dataset path
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
            
        # Initialize HF API
        api = HfApi()
        
        # Create repository
        print(f"\n📦 Creating repository: {repo_name}")
        try:
            create_repo(repo_id=repo_name, repo_type=repo_type)
            print(f"✅ Repository created successfully!")
        except Exception as e:
            print(f"ℹ️ Repository already exists or creation failed: {e}")
        
        # Set default ignore patterns if none provided
        if ignore_patterns is None:
            ignore_patterns = [".git", ".gitignore", "__pycache__", "*.pyc"]
        
        # Upload files
        print(f"\n⬆️ Uploading dataset files to {repo_name}")
        print(f"📁 Uploading from: {dataset_path}")
        
        response = upload_folder(
            folder_path=dataset_path,
            repo_id=repo_name,
            repo_type=repo_type,
            ignore_patterns=ignore_patterns
        )
        
        print("\n✨ Upload completed successfully!")
        print(f"🔗 Dataset available at: https://huggingface.co/datasets/{repo_name}")
        
        # List uploaded files
        print("\n📝 Uploaded files:")
        for file in Path(dataset_path).glob("*"):
            if not any(file.match(pattern) for pattern in ignore_patterns):
                print(f" • {file.name}")
                
        return response
        
    except Exception as e:
        print(f"\n❌ Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hf_dataset_upload.py <dataset_path> <repo_name>")
        print("Example: python hf_dataset_upload.py /path/to/dataset my-username/my-dataset")
        sys.exit(1)
        
    dataset_path = sys.argv[1]
    repo_name = sys.argv[2]
    
    upload_dataset_to_hf(
        dataset_path=dataset_path,
        repo_name=repo_name
    )