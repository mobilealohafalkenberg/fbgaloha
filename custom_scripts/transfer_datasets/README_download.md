# HuggingFace Dataset Download Script

This script allows you to easily download HDF5 datasets from the HuggingFace Hub.

## Prerequisites

Before using the script, ensure you:
1. Have the HuggingFace Hub credentials configured
2. Activate the virtual environment "act"
3. Have sufficient storage space for HDF5 files

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
python hf_dataset_download.py <repo_id> <local_dir>
```

### Data Format
The script will download HDF5 files with the naming pattern `episode_N.hdf5`. The downloaded files will maintain their original structure:
```
/home/aloha/aloha_data/downloaded_toast/
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
python hf_dataset_download.py mobilealohafalkenberg/dataset ~/aloha_data/downloaded_toast

# Using absolute path
python hf_dataset_download.py mobilealohafalkenberg/dataset /home/aloha/aloha_data/downloaded_toast
```

#### If you're in a different directory:
```bash
# Navigate to script directory first
cd ~/custom_scripts/download_fr_huggingface
# OR
cd /home/aloha/custom_scripts/download_fr_huggingface

# Then run the script
python hf_dataset_download.py mobilealohafalkenberg/dataset /home/aloha/aloha_data/downloaded_toast
```

#### Running from anywhere (using absolute paths):
```bash
python /home/aloha/custom_scripts/download_fr_huggingface/hf_dataset_download.py mobilealohafalkenberg/dataset /home/aloha/aloha_data/downloaded_toast
```

## Example Output
```
🤗 Starting HuggingFace Dataset Download Process...
📦 Repository: mobilealohafalkenberg/dataset
📂 Local directory: /home/aloha/aloha_data/downloaded_toast

✨ Download completed successfully!
📁 Files downloaded to: /home/aloha/aloha_data/downloaded_toast

📝 Downloaded files:
 • episode_0.hdf5
 • episode_1.hdf5
 • episode_2.hdf5
 • episode_3.hdf5
 • episode_4.hdf5
 • episode_5.hdf5
 • ...
```

## Note
- Make sure you have sufficient disk space in the target directory before downloading large HDF5 files
- HDF5 files can be large, so ensure stable internet connection during download