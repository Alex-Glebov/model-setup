# Installation Guide

Complete guide for installing model-setup and creating your ML environment.

## Prerequisites

- Python 3.10+
- Linux (Ubuntu 20.04+ recommended)
- Internet connection for downloading packages
- For GPU: NVIDIA or AMD GPU with drivers installed

## Step 1: Clone Repository

```bash
git clone https://github.com/Alex-Glebov/model-setup.git
cd model-setup
```

## Step 2: Run Setup

### Basic Usage

```bash
python test_venv_builder.py /path/to/venv --config /path/to/config.json
```

### Example: Jetson Orin

```bash
python test_venv_builder.py ~/model-core/venv \
    --config ~/model-core/.hardware_config.json
```

This will:
1. Detect Jetson hardware
2. Install PyTorch 2.5.0 from NVIDIA's wheel
3. Install cuSPARSELt 0.7.1.0 locally
4. Install pandas, pyarrow, scikit-learn, tqdm
5. Patch activate script for LD_LIBRARY_PATH

### Example: NVIDIA CUDA

```bash
python test_venv_builder.py ~/myproject/venv \
    --config ~/myproject/.hardware_config.json
```

This will:
1. Detect CUDA GPU
2. Install PyTorch with CUDA 12.1 from PyPI
3. Install remaining dependencies

### Example: CPU Only

```bash
python test_venv_builder.py ~/myproject/venv \
    --config ~/myproject/.hardware_config.json
```

This will:
1. Detect no GPU
2. Install PyTorch CPU version
3. Install remaining dependencies

## Step 3: Activate and Verify

### Option 1: Source Activate

```bash
source ~/model-core/venv/bin/activate
python -c "import torch; print(torch.__version__)"
```

### Option 2: Direct Python Path

```bash
# For Jetson, set LD_LIBRARY_PATH
LD_LIBRARY_PATH=~/model-core/venv/lib/cuda/lib \
    ~/model-core/venv/bin/python -c "import torch; print(torch.__version__)"
```

## Step 4: Install Your Project

If you have a project with additional requirements:

```bash
source ~/model-core/venv/bin/activate
cd ~/your-project
pip install -r requirements.txt
```

## Configuration File Format

The `.hardware_config.json` file contains:

```json
{
  "gpu_type": "jetson",
  "gpu_name": "Orin",
  "compute_capability": "8.7",
  "cuda_version": "12.6",
  "cudnn_version": "9.0"
}
```

## Override Hardware Detection

Create `~/GPU/GPU_VARIANT.txt` to override auto-detection:

```bash
echo "jetson" > ~/GPU/GPU_VARIANT.txt
# or
echo "cuda" > ~/GPU/GPU_VARIANT.txt
# or
echo "rocm" > ~/GPU/GPU_VARIANT.txt
```

## Uninstallation

Simply delete the venv directory:

```bash
rm -rf ~/model-core/venv
```

No system-wide changes were made (no sudo required).

## Next Steps

- [[Troubleshooting]] - If something goes wrong
- [[Hardware Compatibility]] - Check supported GPUs
- [[API Reference]] - For programmatic usage
