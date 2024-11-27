# custom_scripts/download_fr_huggingface/hf_model_download.py:
import os
from huggingface_hub import hf_hub_download, snapshot_download
import sys
from pathlib import Path

def download_model(
    repo_id: str,
    local_dir: str
):
    """
    Download a model from HuggingFace Hub
    
    Args:
        repo_id: HuggingFace repository ID (e.g., 'mobilealohafalkenberg/toast_25_150_2000_1e5_16')
        local_dir: Local directory where to save the model
    """
    try:
        print(f"\n🤗 Starting HuggingFace Download Process...")
        print(f"📦 Repository: {repo_id}")
        print(f"📂 Local directory: {local_dir}")
        
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # Download the complete repository
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False  # Get actual files, not symlinks
        )
        
        print("\n✨ Download completed successfully!")
        print(f"📁 Files downloaded to: {downloaded_path}")
        
        # List downloaded files
        print("\n📝 Downloaded files:")
        for file in Path(downloaded_path).glob("*"):
            print(f"  • {file.name}")
            
        return downloaded_path
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python hf_model_download.py <repo_id> <local_dir>")
        print("Example: python hf_model_download.py mobilealohafalkenberg/toast_25_150_2000_1e5_16 /home/aloha/models/toast")
        sys.exit(1)
        
    repo_id = sys.argv[1]
    local_dir = sys.argv[2]
    
    download_model(repo_id=repo_id, local_dir=local_dir)