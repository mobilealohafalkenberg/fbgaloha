# FBG ALOHA Repository

A modified fork of [Stanford Mobile ALOHA](https://github.com/UT-Austin-RPL/mobile_aloha) / [TrossenRobotics Mobile ALOHA](https://github.com/trossenrobotics/mobile_aloha) configured for:
- Cloud GPU training without physical robot access
- Dataset and model transfer to/from HuggingFace

## Directory Structure

### `act/`
-- start with creating a venv
```bash
python3 -m venv act
# activate the venv "act":
source act/bin/activate
```



### `act_training_evaluation/`
Main project directory containing:
- `act_plus_plus/`: Core training and evaluation code
  - `train.py`: Main script for GPU training in virtual environments (no ROS2/robot dependency)
  - `constants.py`: Task definitions
  - `imitate_episodes.py`: Training script
  - `policy.py`: Policy implementations
  - `visualize_episodes.py`: Visualization utilities
- `robomimic/`: Robotic manipulation framework integration

### `custom_scripts/`
Utility scripts for model and dataset management:
- `models/`: Model management scripts
  - `download_fr_huggingface/`: Download models from HuggingFace
  - `upload_to_huggingface/`: Upload models to HuggingFace
- `transfer_datasets/`: Dataset transfer utilities
  - `hf_data_download.py`: Download training datasets from HuggingFace
  - `hf_data_upload.py`: Upload datasets to HuggingFace

## Workflow
1. Download training data using `hf_data_download.py`
2. Train models using `train.py` on GPU-enabled virtual machine
3. Upload trained models using scripts in `models/upload_to_huggingface/`

## Setup
1. Activate the virtual environment:
```bash
source act/bin/activate
```
2. The activated environment provides necessary dependencies for running training and evaluation scripts.

Note: Additional setup or configuration steps may be required. Please consult project maintainers for complete setup instructions.