# HuggingFace Model Upload Script

A Python script to easily upload trained models to the HuggingFace Hub using cached credentials.

## Prerequisites

- Python 3.10
- Virtual environment (located at `/home/aloha/act`)
- HuggingFace account (already logged in)

## Setup

Activate the virtual environment using the absolute path (this works from any directory):
```bash
source /home/aloha/act/bin/activate
# or if in aloha directory:
source act/bin/activate
```

## CLI Usage

```bash
# Basic usage
python hf_model_upload.py <model_path> <repo_name>

# Example with actual paths
python hf_model_upload.py /home/user/projects/my_model mobilealohafalkenberg/model-name

# Examples
# Upload a local BERT model
python hf_model_upload.py /home/user/models/bert-custom mobilealohafalkenberg/bert-custom


# Upload from current directory
python hf_model_upload.py ./my_model_dir username/my-model
```

## Features

- Automatically creates a new repository if it doesn't exist
- Uploads all files from the specified model directory
- Provides progress feedback during upload
- Returns the upload response for verification
- Displays the HuggingFace Hub URL after successful upload

## Output Indicators

The script uses emoji indicators to show progress:
- 🤗 : Start of upload process
- 📦 : Repository creation
- ⬆️  : File upload in progress
- ✅ : Successful completion
- ❌ : Error occurred
- 🔗 : Model URL

## Error Handling

The script includes comprehensive error handling and will:
- Inform if the repository already exists
- Display detailed error messages if upload fails
- Raise exceptions with specific error information

## Note

Make sure you have:
1. A stable internet connection for the upload process
2. Sufficient permissions to create repositories and upload models
3. The correct repository name format (username/repository-name)
