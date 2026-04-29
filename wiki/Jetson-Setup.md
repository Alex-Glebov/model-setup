# Jetson Setup Guide

Complete guide for setting up PyTorch on NVIDIA Jetson devices.

## Overview

Jetson devices (Orin Nano, Orin NX, Orin AGX) require special handling due to their ARM64 architecture and integrated GPU.

## Prerequisites

### JetPack Version

Check your JetPack version:

```bash
cat /etc/nv_tegra_release
# R36 (release), REVISION: 2.0 = JetPack 6.0
# R35 (release), REVISION: 4.1 = JetPack 5.1
```

**Supported**: JetPack 6.x only (R36.x)

### System Requirements

- JetPack 6.0 or 6.1
- Python 3.10
- 4GB+ free space
- Internet connection

## Quick Setup

```bash
# Clone model-setup
git clone https://github.com/Alex-Glebov/model-setup.git
cd model-setup

# Create venv
python model-setup ~/model-core/venv \
    --config ~/model-core/.hardware_config.json

# Activate and verify
source ~/model-core/venv/bin/activate
python -c "import torch; print(f'PyTorch {torch.__version__}'); \
    print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## What Gets Installed

### PyTorch 2.5.0

From NVIDIA's JetPack repository:
```
https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/
torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

This wheel is:
- Compiled for sm_87 (Orin compute capability)
- Includes CUDA 12.6 support
- Bundles numpy 1.x, sympy, jinja2

### cuSPARSELt 0.7.1.0

**Required for**: PyTorch 2.4+

**Installed to**: `venv/lib/cuda/lib/`

Libraries:
- `libcusparseLt.so`
- `libcusparseLt.so.0`
- `libcusparseLt.so.0.7.1.0`

### LD_LIBRARY_PATH

The activate script is patched to:
```bash
export LD_LIBRARY_PATH="/path/to/venv/lib/cuda/lib:$LD_LIBRARY_PATH"
```

This ensures PyTorch can find cuSPARSELt at runtime.

## Manual Installation (if needed)

### Step 1: Create Venv

```bash
python3 -m venv ~/model-core/venv
```

### Step 2: Install PyTorch

```bash
source ~/model-core/venv/bin/activate
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/\
torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

### Step 3: Install cuSPARSELt

```bash
# Download
cd /tmp
wget https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/\
linux-aarch64/libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz

# Extract
tar -xf libcusparse_lt-linux-aarch64-0.7.1.0-archive.tar.xz

# Copy to venv
mkdir -p ~/model-core/venv/lib/cuda/lib
cp libcusparse_lt-linux-aarch64-0.7.1.0-archive/lib/*.so* \
    ~/model-core/venv/lib/cuda/lib/
```

### Step 4: Patch Activate Script

Add to `~/model-core/venv/bin/activate`:
```bash
_OLD_VIRTUAL_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -d "$VIRTUAL_ENV/lib/cuda/lib" ] ; then
    LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LD_LIBRARY_PATH
fi
```

### Step 5: Install Other Packages

```bash
source ~/model-core/venv/bin/activate
pip install pandas pyarrow scikit-learn tqdm
```

## Verification

### Basic Check

```bash
source ~/model-core/venv/bin/activate
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

Expected output:
```
PyTorch: 2.5.0a0+872d972e41.nv24.08
CUDA available: True
Device: Orin
Memory: 7.4 GB
```

### Performance Test

```bash
source ~/model-core/venv/bin/activate
cd ~/model-core
python test_quick.py
```

Expected: ~1.2 seconds per epoch (proof-of-concept mode)

## Common Issues

### "libcusparseLt.so.0: cannot open shared object file"

**Cause**: LD_LIBRARY_PATH not set.

**Fix**:
```bash
source ~/model-core/venv/bin/activate
# Or manually:
export LD_LIBRARY_PATH=~/model-core/venv/lib/cuda/lib:$LD_LIBRARY_PATH
```

### "no kernel image is available"

**Cause**: Using standard PyTorch wheel (sm_80/sm_90) instead of Jetson wheel (sm_87).

**Fix**: Reinstall with NVIDIA's Jetson wheel:
```bash
pip uninstall torch
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/\
torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

### NumPy version errors

**Cause**: Package upgraded numpy to 2.x, but PyTorch needs 1.x.

**Fix**:
```bash
pip install 'numpy<2'
```

## Memory Optimization

Jetson Orin Nano has 8GB shared memory. For large models:

1. **Create swap**:
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

2. **Use proof-of-concept mode**:
```python
model = LSTMAttentionModel(
    sequence_length=50,
    n_features=9,
    proof_of_concept=True  # Smaller model
)
```

3. **Reduce batch size**:
```python
model.train(..., batch_size=8)  # Instead of 32
```

## JetPack 5.x Notes

JetPack 5.x is **not supported** by model-setup. If you must use it:

- Use PyTorch 2.1.0
- No cuSPARSELt required
- Different wheel URL

See NVIDIA's JetPack 5.1 compatibility matrix.

## Next Steps

- [[Troubleshooting]] - More issues and solutions
- [[Hardware Compatibility]] - Full compatibility matrix
- [model-core documentation](https://github.com/Alex-Glebov/model-core) - Using the venv
