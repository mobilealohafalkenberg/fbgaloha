# # custom_scripts/upload_to_huggingface/hf_model_upload.py:
import os
from huggingface_hub import HfApi, create_repo, upload_folder
import sys

def upload_model_to_hf(
    model_path: str,
    repo_name: str,
    repo_type: str = "model",
):
    """
    Upload a trained model to HuggingFace Hub using cached credentials
    """
    try:
        print("\n🤗 Starting HuggingFace Upload Process...")
        
        # Initialize HF API
        api = HfApi()
        
        # Create repository
        print(f"\n📦 Creating repository: {repo_name}")
        try:
            create_repo(repo_id=repo_name, repo_type=repo_type)
            print(f"✅ Repository created successfully!")
        except Exception as e:
            print(f"ℹ️  Repository already exists or creation failed: {e}")
        
        # Upload files
        print(f"\n⬆️  Uploading model files to {repo_name}")
        response = upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            repo_type=repo_type
        )
        
        print("\n✨ Upload completed successfully!")
        print(f"🔗 Model available at: https://huggingface.co/{repo_name}")
        
        return response
        
    except Exception as e:
        print(f"\n❌ Error uploading model: {e}")
        raise

if __name__ == "__main__":
    model_path = sys.argv[1]
    repo_name = sys.argv[2]
    
    upload_model_to_hf(
        model_path=model_path,
        repo_name=repo_name
    )