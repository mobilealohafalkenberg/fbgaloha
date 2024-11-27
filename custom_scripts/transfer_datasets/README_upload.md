# HuggingFace Dataset Upload Script

This script allows you to easily upload HDF5 datasets to the HuggingFace Hub.

## Prerequisites

Before using the script, ensure you:
1. Have the HuggingFace Hub credentials configured
2. Activate the virtual environment "act"
3. Have your HDF5 (*.hdf5) files ready for upload

## Environment Setup

Activate the virtual environment:
```bash
# If you're in the home directory
source ~/act/bin/activate  # or
source /home/aloha/act/bin/activate

# If you're in the act directory
source ./bin/activate
```

## Usage

### Command Format
```bash
python hf_dataset_upload.py <dataset_path> <repo_name>
```

### Data Format
The script expects HDF5 files with the naming pattern `episode_N.hdf5`. Example structure:
```
/home/aloha/aloha_data/toast/
├── episode_0.hdf5
├── episode_1.hdf5
├── episode_2.hdf5
├── episode_3.hdf5
├── episode_4.hdf5
├── episode_5.hdf5
└── ...
```

### Examples

#### If you're in the directory containing the script:
```bash
# Using relative path
python hf_dataset_upload.py ~/aloha_data/toast mobilealohafalkenberg/dataset

# Using absolute path
python hf_dataset_upload.py /home/aloha/aloha_data/toast mobilealohafalkenberg/dataset
```

#### If you're in a different directory:
```bash
# Navigate to script directory first
cd ~/custom_scripts/upload_to_huggingface
# OR
cd /home/aloha/custom_scripts/upload_to_huggingface

# Then run the script
python hf_dataset_upload.py /home/aloha/aloha_data/toast mobilealohafalkenberg/dataset
```

#### Running from anywhere (using absolute paths):
```bash
python /home/aloha/custom_scripts/upload_to_huggingface/hf_dataset_upload.py /home/aloha/aloha_data/toast mobilealohafalkenberg/dataset
```

## Example Output
```
🤗 Starting HuggingFace Dataset Upload Process...

📦 Creating repository: mobilealohafalkenberg/dataset
✅ Repository created successfully!

⬆️ Uploading dataset files to mobilealohafalkenberg/dataset
📁 Uploading from: /home/aloha/aloha_data/toast

📝 Uploaded files:
 • episode_0.hdf5
 • episode_1.hdf5
 • episode_2.hdf5
 • episode_3.hdf5
 • episode_4.hdf5
 • episode_5.hdf5
 • ...

✨ Upload completed successfully!
🔗 Dataset available at: https://huggingface.co/datasets/mobilealohafalkenberg/dataset
```