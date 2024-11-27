# HuggingFace Model Downloader

A simple utility script to download models from HuggingFace Hub with progress tracking and error handling.

## Setup

1. Clone this repository or download the script
2. Activate the Python virtual environment:
```bash
# If you're in /home/aloha directory:
source act/bin/activate

# Or using full path (can be run from anywhere):
source /home/aloha/act/bin/activate
```

3. Install the required dependencies in the activated environment:
```bash
pip install huggingface-hub
```

## Features

- Downloads complete model repositories from HuggingFace Hub
- Creates local directory structure automatically
- Shows download progress with emoji-based status indicators
- Lists all downloaded files after completion
- Supports both CLI and programmatic usage
- Error handling with informative messages

## Usage

### Command Line Interface

Make sure the virtual environment is activated first, then:

```bash
python download_model.py <repo_id> <local_dir>
```

Example:
```bash
# Activate environment if not already activated
source /home/aloha/act/bin/activate

# Run the script
python hf_model_download.py mobilealohafalkenberg/toast_25_150_2000_1e5_16 /home/aloha/models/toast
# or, for any other model repo
python hf_model_download.py mobilealohafalkenberg/sort_garbage /home/aloha/models/sort_garbage
```

### Programmatic Usage

```python
# Make sure to run this in the activated virtual environment
from hf_model_download import download_model

# Download a model
downloaded_path = download_model(
    repo_id="mobilealohafalkenberg/toast_25_150_2000_1e5_16",
    local_dir="/home/aloha/models/toast"
)
```

## Parameters

- `repo_id` (str): The HuggingFace repository ID (e.g., 'mobilealohafalkenberg/toast_25_150_2000_1e5_16')
- `local_dir` (str): Local directory where the model should be saved

## Output

The script provides visual feedback during the download process:

- 🤗 Indicates start of download process
- 📦 Shows repository being downloaded
- 📂 Displays local download directory
- ✨ Confirms successful download
- 📁 Shows final download location
- 📝 Lists all downloaded files

## Error Handling

If an error occurs during download, the script will:
1. Print an error message with the ❌ indicator
2. Show the specific error details
3. Raise the exception for proper error handling

## Dependencies

- Python 3.6+
- huggingface_hub
- pathlib
- os
- sys

## Virtual Environment

This project uses a Python virtual environment named "act" located at `/home/aloha/act`. Always ensure you're working within this environment:

```bash
# Check if you're in the correct environment
which python
# Should show: /home/aloha/act/bin/python

# If not in the environment, activate it
source /home/aloha/act/bin/activate
```

To deactivate the environment when you're done:
```bash
deactivate
```


## Acknowledgments

This script uses the HuggingFace Hub library for downloading models. Visit [HuggingFace Hub](https://huggingface.co/) for more information about available models.