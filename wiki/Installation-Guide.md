# Installation Guide

Complete guide for installing model-setup and creating your ML environment.

## Prerequisites

- Python 3.10+
- Linux (Ubuntu 20.04+ recommended) or Windows Subsystem for Linux (WSL 2)
- Internet connection for downloading packages
- For GPU: NVIDIA or AMD GPU with drivers installed (see platform-specific sections)

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
    --config ~/model-core/hardware_config.json \
    --log-file ~/jetson-setup.log
```

This will:
1. Detect Jetson hardware and JetPack version
2. Install PyTorch from NVIDIA's wheel (matched to JetPack)
3. Install cuSPARSELt locally (auto-detected)
4. Install remaining dependencies
5. Patch activate script for LD_LIBRARY_PATH

### Example: NVIDIA CUDA

```bash
python test_venv_builder.py ~/myproject/venv \
    --config ~/myproject/hardware_config.json \
    --log-file ~/cuda-setup.log
```

This will:
1. Detect CUDA GPU and version
2. Install PyTorch with matching CUDA version from PyPI (e.g., cu121, cu124)
3. Install remaining dependencies

### Example: CPU Only

```bash
python test_venv_builder.py ~/myproject/venv \
    --config ~/myproject/hardware_config.json
```

This will:
1. Detect no GPU
2. Install PyTorch CPU version
3. Install remaining dependencies

### Example: Install All Backends

```bash
python test_venv_builder.py ~/myproject/venv \
    --config ~/myproject/hardware_config.json \
    --all
```

This will:
1. Detect CUDA GPU
2. Install both torch (CUDA) and tensorflow (CUDA)
3. Generate `model_core/keras_backend.py` with commented switch options

Switch backends later by editing `keras_backend.py`:
```python
# Comment out torch
# os.environ["KERAS_BACKEND"] = "torch"

# Uncomment tensorflow
os.environ["KERAS_BACKEND"] = "tensorflow"
```

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

## WSL (Windows) Installation

### Prerequisites

1. **Windows 10/11 with WSL 2**
2. **NVIDIA GPU drivers** installed on Windows host
3. **WSL 2** (not WSL 1 - GPU passthrough requires WSL 2)

### Setup Steps

1. **Install WSL 2:**
   ```powershell
   # In PowerShell as Administrator
   wsl --install
   # Or if already installed:
   wsl --update
   ```

2. **Install NVIDIA drivers on Windows:**
   - Download from [NVIDIA drivers](https://www.nvidia.com/drivers/)
   - Install on Windows (not in WSL)

3. **Verify WSL GPU support:**
   ```bash
   # In WSL terminal
   ls /usr/lib/wsl/lib/
   # Should see: libcuda.so, libd3d12.so, libdxcore.so
   ```

4. **Run model-setup:**
   ```bash
   cd model-setup
   python test_venv_builder.py ~/venv \
       --config ~/hardware_config.json \
       --log-file ~/wsl-setup.log
   ```

See [[WSL Support]] for detailed troubleshooting.

## Configuration File Format

The `hardware_config.json` file contains:

```json
{
  "platform": "Linux",
  "machine": "x86_64",
  "gpu_type": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 4060",
  "gpu_memory_mb": 8188,
  "cuda_version": "12.2",
  "cudnn_version": null,
  "compute_capability": null,
  "preferred_backend": "pytorch",
  "gpu_available": true,
  "is_wsl": false,
  "keras_backend": "torch",
  "available_backends": ["torch", "tensorflow"]
}
```

When using `--all`, `available_backends` lists all successfully installed backends.

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
