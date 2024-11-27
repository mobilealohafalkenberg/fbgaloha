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
3. Install required dependencies:
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
```bash
python hf_model_download.py <repo_id> <local_dir>

# Example:
source /home/aloha/act/bin/activate
python hf_model_download.py mobilealohafalkenberg/toast_25_150_2000_1e5_16 /home/aloha/models/toast
```

### Programmatic Usage
```python
from hf_model_download import download_model

downloaded_path = download_model(
    repo_id="mobilealohafalkenberg/toast_25_150_2000_1e5_16",
    local_dir="/home/aloha/models/toast"
)
```

## Parameters
- `repo_id` (str): The HuggingFace repository ID
- `local_dir` (str): Local directory where the model should be saved

## Output Indicators
- 🤗 Start of download process
- 📦 Repository download
- 📂 Local download directory
- ✨ Successful download
- 📁 Final download location
- 📝 Downloaded files list

## Error Handling
The script will:
1. Print error message with ❌ indicator
2. Show specific error details
3. Raise exception for proper handling

## Dependencies
- Python 3.6+
- huggingface_hub
- pathlib
- os
- sys

## Virtual Environment
Project uses Python virtual environment at `/home/aloha/act`:
```bash
# Verify environment
which python
# Should show: /home/aloha/act/bin/python

# Activate if needed
source /home/aloha/act/bin/activate

# Deactivate when done
deactivate
```

## Acknowledgments
Uses HuggingFace Hub library for downloading models. Visit [HuggingFace Hub](https://huggingface.co/) for more information.