# HuggingFace Model Upload Script
A Python script to easily upload trained models to the HuggingFace Hub using cached credentials.

## Prerequisites
- Python 3.10
- Virtual environment (at `/home/aloha/act`)
- HuggingFace account (already logged in)

## Setup
```bash
# Activate from any directory
source /home/aloha/act/bin/activate
# or if in aloha directory:
source act/bin/activate
```

## CLI Usage
```bash
# Basic usage
python hf_model_upload.py <model_path> <repo_name>

# Examples
python hf_model_upload.py /home/user/models/bert-custom mobilealohafalkenberg/bert-custom
python hf_model_upload.py ./my_model_dir username/my-model
```

## Features
- Auto-creates new repositories
- Uploads all files from model directory
- Provides progress feedback
- Returns upload response
- Shows HuggingFace Hub URL after success

## Output Indicators
- 🤗 Start of upload
- 📦 Repository creation
- ⬆️ Upload in progress
- ✅ Successful completion
- ❌ Error occurred
- 🔗 Model URL

## Error Handling
The script handles:
- Existing repository checks
- Detailed error messages
- Specific error information in exceptions

## Requirements
1. Stable internet connection
2. Repository creation/upload permissions
3. Correct repository name format (username/repository-name)