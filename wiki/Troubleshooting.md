# Troubleshooting

Common issues and solutions for model-setup.

## Table of Contents

1. [Checking Logs](#checking-logs)
2. [WSL Issues](#wsl-issues)
3. [GPU Prerequisites Missing](#gpu-prerequisites-missing)
4. [Import Errors](#import-errors)
5. [GPU Not Detected](#gpu-not-detected)
6. [Library Loading Errors](#library-loading-errors)
7. [Version Conflicts](#version-conflicts)
8. [Installation Failures](#installation-failures)

## Checking Logs

model-setup logs detailed information about what's happening. Always check logs first:

### Using model-setup

```bash
# Default log location
cat ~/model-core/test_venv_builder.log

# Or with custom log
cat ~/my-setup.log
```

### Using module directly

```bash
cat ~/model-core/venv/install.log
```

### Filtering for errors

```bash
# See all warnings and errors
cat test_venv_builder.log | grep -E "(WARNING|ERROR|prerequisites|Missing)"

# See hardware detection
cat test_venv_builder.log | grep -E "(Detected|hardware|WSL)"

# See installation steps
cat test_venv_builder.log | grep -E "(Installing|installed|PASSED|FAILED)"
```

## WSL Issues

### "WSL 1 detected - WSL 2 required for GPU support"

**Cause**: Running WSL 1, which doesn't support GPU passthrough.

**Solution**: Upgrade to WSL 2:
```powershell
# In PowerShell as Administrator
wsl --set-version Ubuntu 2
# Or set default
wsl --set-default-version 2
```

### "WSL GPU support not detected (/usr/lib/wsl/lib missing)"

**Cause**: GPU drivers not installed on Windows host.

**Solution**:
1. Install NVIDIA drivers on Windows (not in WSL): https://www.nvidia.com/drivers/
2. Reboot Windows
3. Verify in WSL:
   ```bash
   ls /usr/lib/wsl/lib/ | grep cuda
   ```

### "WSL DirectX support missing (libdxcore.so not found)"

**Cause**: WSL not fully updated or GPU not properly passed through.

**Solution**:
```powershell
# In PowerShell as Administrator
wsl --update
# Then restart WSL
wsl --shutdown
```

## GPU Prerequisites Missing

### "nvidia-smi not found - NVIDIA drivers not installed"

**Cause**: NVIDIA GPU drivers not installed on the system.

**Solution**:
- **Linux**: Install from [NVIDIA drivers](https://www.nvidia.com/drivers/)
- **WSL**: Install on Windows host (not in WSL), then restart WSL

> **Note**: The CUDA toolkit (nvcc) and cuDNN are NOT prerequisites - the
> PyTorch/TensorFlow wheels bundle the CUDA runtime and cuDNN. A working
> `nvidia-smi` (driver) is all model-setup checks for.

### "ROCm not installed (rocm-smi not found)"

**Cause**: AMD ROCm not installed.

**Solution**: Install ROCm: https://www.amd.com/en/developer/rocm-hub.html

### Understanding the Messages

When model-setup detects a missing driver, it will still attempt to install but warns you:

```
WARNING - CUDA prerequisites missing:
WARNING -   - nvidia-smi not found - NVIDIA drivers not installed
INFO - Will attempt install anyway, but may fail
```

This means:
- The installation **may still work** if the Python packages include their own CUDA libraries
- But if it fails, you know exactly what to install
- After installing missing components, just run the script again

## Import Errors

### "No module named 'torch'"

**Cause**: PyTorch installation failed or venv not activated.

**Solution**:
```bash
# Check if PyTorch is installed
/path/to/venv/bin/pip list | grep torch

# If missing, reinstall
/path/to/venv/bin/pip install torch
```

### "No module named 'model_setup'"

**Cause**: Running from outside project directory.

**Solution**:
```bash
cd /path/to/model-setup
python model-setup ...
```

## GPU Not Detected

### "CUDA available: False" on Jetson

**Cause**: Missing cuSPARSELt or LD_LIBRARY_PATH not set.

**Solution**:
```bash
# Check cuSPARSELt exists
ls /path/to/venv/lib/cuda/lib/libcusparseLt.so*

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/venv/lib/cuda/lib:$LD_LIBRARY_PATH

# Or source activate
source /path/to/venv/bin/activate
```

### Wrong GPU detected

**Cause**: Multiple GPUs or detection error.

**Solution**: Override with explicit file:
```bash
mkdir -p ~/GPU
echo "jetson" > ~/GPU/GPU_VARIANT.txt
```

## Library Loading Errors

### "libcusparseLt.so.0: cannot open shared object file"

**Jetson-specific**: cuSPARSELt is required for PyTorch 2.4+ but not found.

**Solution 1: Source activate** (recommended)
```bash
source ~/model-core/venv/bin/activate
python your_script.py
```

**Solution 2: Set LD_LIBRARY_PATH**
```bash
LD_LIBRARY_PATH=~/model-core/venv/lib/cuda/lib \
    python your_script.py
```

**Solution 3: Direct Python path**
```bash
LD_LIBRARY_PATH=~/model-core/venv/lib/cuda/lib \
    ~/model-core/venv/bin/python your_script.py
```

### "libcudnn.so.8: cannot open shared object file"

**Cause**: PyTorch compiled for cuDNN 8, system has cuDNN 9.

**Solution**: On JetPack 6.x, use NVIDIA's PyTorch wheel (model-setup does this automatically).

## Version Conflicts

### "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"

**Cause**: An old PyTorch build (< 2.3) compiled against numpy 1.x, but another
package upgraded numpy to 2.x.

**Solution**: model-setup installs current PyTorch (2.3+), which supports
numpy 2.x, so this should not occur in model-setup venvs. If you must keep an
old PyTorch, pin numpy manually:
```bash
pip install 'numpy<2'
```

### "CUDA error: no kernel image is available"

**Cause**: PyTorch compiled for wrong compute capability.

**Solution**: Use NVIDIA's wheel for Jetson:
```bash
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/\
torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

## Installation Failures

### "Could not find a version that satisfies the requirement"

**Cause**: Wrong Python version or platform.

**Check**:
```bash
python --version  # Should be 3.10
uname -m          # Should be aarch64 for Jetson
```

### JetPack version not detected

**Cause**: `/etc/nv_tegra_release` not found or unreadable.

**Manual override**:
```python
# In venv_builder.py or set environment variable
export JETPACK_VERSION=6.0
```

### Download timeouts

**Cause**: Slow internet or large package (PyTorch wheel is 800MB).

**Solution**: Use cached wheel or download manually:
```bash
# Download first
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/\
torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# Install from file
pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

### "Unknown install type: torch" or "Unknown install type: tensorflow"

**Cause**: Bug in older version of model-setup.

**Solution**: Update to latest version:
```bash
git pull origin develop
```

This was fixed in commit `af69671` - the `_build_install_queue()` was returning backend names instead of install types.

## Getting Help

If your issue isn't listed here:

1. Check [GitHub Issues](https://github.com/Alex-Glebov/model-setup/issues)
2. Include in your report:
   - Hardware: `cat /etc/nv_tegra_release` (Jetson) or `nvidia-smi` (CUDA)
   - Python version: `python --version`
   - Error message: Full traceback
   - Steps to reproduce
