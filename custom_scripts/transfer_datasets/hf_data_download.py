import os
from huggingface_hub import hf_hub_download, snapshot_download
import sys
from pathlib import Path
from typing import Optional, List

def download_dataset(
    repo_id: str,
    local_dir: str,
    revision: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None
):
    """
    Download a dataset from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        local_dir: Local directory where to save the dataset
        revision: Optional git revision to download (branch name, commit hash)
        ignore_patterns: List of file patterns to ignore during download
    """
    try:
        print(f"\n🤗 Starting HuggingFace Dataset Download Process...")
        print(f"📦 Repository: {repo_id}")
        print(f"📂 Local directory: {local_dir}")
        
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Set default ignore patterns if none provided
        if ignore_patterns is None:
            ignore_patterns = [".git", ".gitignore", "__pycache__", "*.pyc"]
        
        # Download the complete repository
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            ignore_patterns=ignore_patterns,
            revision=revision,
            repo_type="dataset",
            local_dir_use_symlinks=False  # Get actual files, not symlinks
        )
        
        print("\n✨ Download completed successfully!")
        print(f"📁 Files downloaded to: {downloaded_path}")
        
        # List downloaded files
        print("\n📝 Downloaded files:")
        for file in Path(downloaded_path).glob("*"):
            if not any(file.match(pattern) for pattern in ignore_patterns):
                print(f" • {file.name}")
                
        return downloaded_path
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hf_dataset_download.py <repo_id> <local_dir>")
        print("Example: python hf_dataset_download.py username/dataset-name /path/to/local/dir")
        sys.exit(1)
        
    repo_id = sys.argv[1]
    local_dir = sys.argv[2]
    
    download_dataset(
        repo_id=repo_id,
        local_dir=local_dir
    )