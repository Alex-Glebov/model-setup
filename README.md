# Model Setup

Automated hardware detection and virtual environment setup for ML training. Cross-platform support for Linux, Windows+WSL, NVIDIA Jetson, CUDA, ROCm, and CPU-only systems.

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20WSL%20%7C%20aarch64%20%7C%20x86_64-blue)](https://github.com/Alex-Glebov/model-setup)
[![GPU](https://img.shields.io/badge/GPU-Jetson%20%7C%20CUDA%20%7C%20ROCm%20%7C%20CPU-green.svg)]()

## Features

- **Hardware Auto-Detection**: Automatically detects GPU type (Jetson/CUDA/ROCm/CPU) and version
- **Dynamic Version Selection**: Automatically matches detected CUDA/ROCm version to appropriate PyTorch wheel (no hardcoded versions)
- **Cross-Platform**: Linux native + Windows Subsystem for Linux (WSL) support
- **WSL Detection**: Automatically detects WSL environment and checks WSL 2 + GPU prerequisites
- **Detailed Diagnostics**: Logs exactly why GPU setup fails (missing drivers, toolkit, libraries)
- **Multi-Backend Support**: Install all working backends (`--all` flag) with easy switching
- **Jetson Optimized**: Full support for Jetson Orin with sm_87, cuSPARSELt
- **No Sudo Required**: All dependencies installed locally in venv
- **Keras 3.x Ready**: Supports torch, tensorflow, and jax backends

## Quick Start

```bash
git clone https://github.com/Alex-Glebov/model-setup.git
cd model-setup
python test_venv_builder.py /path/to/venv --config /path/to/hardware_config.json
```

Example:
```bash
python test_venv_builder.py ~/model-core/venv --config ~/model-core/hardware_config.json
```

### Install All Working Backends

```bash
python test_venv_builder.py ~/model-core/venv --all --config ~/model-core/hardware_config.json
```

This installs all working backends (e.g., torch CUDA + tensorflow CUDA) and generates `keras_backend.py` with commented lines for easy switching.

### With Custom Log File

```bash
python test_venv_builder.py ~/model-core/venv --log-file ~/setup.log --config ~/model-core/hardware_config.json
```

## Hardware Support

| Platform | Status | PyTorch | Notes |
|----------|--------|---------|-------|
| NVIDIA Jetson Orin | ✅ Supported | 2.5.0+ (NVIDIA wheel) | sm_87, cuSPARSELt auto-detected |
| NVIDIA CUDA | ✅ Supported | 2.x (PyPI) | Auto-detects CUDA 11.8, 12.1, 12.4, 12.8+ |
| AMD ROCm | ✅ Supported | 2.x (PyPI) | Auto-detects ROCm 5.6, 5.7, 6.0+ |
| CPU Only | ✅ Supported | 2.x (PyPI) | No GPU acceleration |
| Windows WSL | ✅ Supported | 2.x (PyPI) | WSL 2 required for GPU passthrough |

### Multi-Backend Support

Install multiple backends and switch between them:

```bash
python test_venv_builder.py ~/venv --all
```

Generates `model_core/keras_backend.py`:
```python
# Active backend: torch
os.environ["KERAS_BACKEND"] = "torch"

# Alternative backends (uncomment to switch):
# Backend: tensorflow (cuda)
# os.environ["KERAS_BACKEND"] = "tensorflow"
```

### Why PyTorch over TensorFlow?

- **Jetson**: NVIDIA provides official PyTorch wheels for JetPack 6.x; TensorFlow GPU support is not officially stable
- **Performance**: Better sm_87 (Orin) compute capability support
- **Installation**: PyTorch wheels include all dependencies; TensorFlow requires manual dependency management
- **Keras 3.x**: PyTorch is the default Keras backend with best compatibility

## Architecture

```
model-setup (this project)
    ├── Detects hardware (hardware_detector.py)
    ├── Installs dependencies (venv_builder.py)
    ├── Creates venv with PyTorch
    └── Writes .hardware_config.json
           ↓
    model-core (runtime)
        ├── Reads .hardware_config.json
        ├── Configures PyTorch/Torch device
        └── Runs training/inference
```

## Installation Details

### What Gets Installed

1. **PyTorch** - Hardware-specific version with auto-detection:
   - **Jetson**: NVIDIA's wheel from `developer.download.nvidia.com` (matches JetPack version)
   - **CUDA**: `torch` from PyPI with detected CUDA version (cu118, cu121, cu124, cu128+)
   - **ROCm**: `torch` from PyPI with detected ROCm version (rocm5.6, rocm5.7, rocm6.0+)
   - **CPU**: `torch` from PyPI CPU-only

2. **cuSPARSELt** (Jetson only):
   - Auto-detected and installed locally in `venv/lib/cuda/lib/`
   - No system-wide changes (no sudo)

3. **NumPy**:
   - Version < 2.0 (pinned for PyTorch compatibility)
   - Bundled with PyTorch wheel

4. **Other Dependencies**:
   - Read from `model-core/requirements.txt`
   - Skips packages already provided by PyTorch

### Version Detection

The builder automatically detects and matches versions:

```python
# Detected CUDA 12.2 → Uses PyTorch cu122 wheel
# Detected ROCm 5.7 → Uses PyTorch rocm5.7 wheel
# Detected JetPack 6.0 → Uses NVIDIA JetPack 6.x wheel
```

If detection fails, falls back to latest stable (cu121 for CUDA, rocm5.7 for ROCm).

### Using the Venv

**Option 1: Source activate (recommended)**
```bash
cd ~/model-core
source venv/bin/activate
python train.py
```

**Option 2: Direct Python path**
```bash
# For Jetson, LD_LIBRARY_PATH must include cuSPARSELt
LD_LIBRARY_PATH=~/model-core/venv/lib/cuda/lib \
    ~/model-core/venv/bin/python train.py
```

## Verification

```python
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
```

## Project Structure

```
model-setup/
├── src/model_setup/
│   ├── __init__.py
│   ├── hardware_detector.py    # Hardware detection (GPU type, versions, WSL)
│   ├── venv_builder.py         # Venv creation with dynamic version selection
│   ├── gpu_compatibility.py    # GPU compatibility testing
│   ├── pip_version_checker.py  # PyPI availability checking
│   └── version_requirements.py # Minimum version requirements
├── test_venv_builder.py       # CLI entry point with file logging
└── README.md                   # This file
```

## WSL Support (Windows Subsystem for Linux)

model-setup supports WSL 2 for GPU passthrough on Windows.

### Prerequisites

- **WSL 2** (WSL 1 does not support GPU)
- **GPU drivers** installed on Windows host
- **WSL GPU support** (`/usr/lib/wsl/lib` must exist)

### Automatic Detection

The builder automatically detects WSL and checks prerequisites:

```
INFO - WSL environment detected
INFO - WSL 2 confirmed
WARNING - WSL GPU support not detected (/usr/lib/wsl/lib missing)
```

### Manual WSL Check

```bash
# Verify WSL version
wsl.exe --version

# Should show: WSL version: 2.x.x

# Check GPU in WSL
ls /usr/lib/wsl/lib/ | grep -i cuda
# Should show: libcuda.so, libd3d12.so, etc.
```

## Detailed Diagnostics

model-setup logs exactly why GPU setup fails. Check the log file for details:

### CUDA Prerequisites Check

```
INFO - Detecting hardware on Linux x86_64
INFO - WSL environment detected
INFO - CUDA GPU detected
WARNING - CUDA prerequisites missing:
WARNING -   - nvidia-smi not found - NVIDIA drivers not installed
WARNING -   - CUDA toolkit not installed (nvcc not found)
WARNING -   - cuDNN library (optional but recommended)
INFO - Will attempt install anyway, but may fail
```

### ROCm Prerequisites Check

```
WARNING - ROCm prerequisites missing:
WARNING -   - ROCm not installed (rocm-smi not found)
WARNING -   - HIP toolkit not installed (hipcc not found)
```

### Log File Locations

**Using test_venv_builder.py:**
- Default: `{venv_parent}/test_venv_builder.log`
- Custom: `--log-file ~/custom.log`

**Using module directly:**
- `{venv_path}/install.log`

## Configuration Priority

1. **Explicit** (highest): `~/GPU/GPU_VARIANT.txt` containing `jetson`, `cuda`, or `rocm`
2. **Auto-detect**: PyTorch CUDA availability check
3. **Platform inference**: `aarch64` → likely Jetson

## Troubleshooting

### GPU Not Detected / Prerequisites Missing

Check the log file for exact reason:
```bash
cat ~/model-core/test_venv_builder.log | grep -E "(prerequisites|Missing|not found)"
```

Common issues:
- **NVIDIA drivers**: Install from [NVIDIA drivers](https://www.nvidia.com/drivers/)
- **CUDA toolkit**: Install from [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- **ROCm**: Install from [AMD ROCm](https://www.amd.com/en/developer/rocm-hub.html)

### WSL GPU Issues

```bash
# Verify WSL 2
wsl.exe --version
# Expected: WSL version: 2.x.x

# Check Windows GPU drivers
# Install from NVIDIA/AMD website on Windows host

# Verify WSL GPU passthrough
ls -la /usr/lib/wsl/lib/
# Should see: libcuda.so, libd3d12.so, libdxcore.so
```

### "libcusparseLt.so.0: cannot open shared object file" (Jetson only)
- Venv not activated: Run `source venv/bin/activate`
- Or missing LD_LIBRARY_PATH when using direct Python path

### "No module named 'torch'"
- Venv creation failed - check hardware detection output in log file
- Try running test_venv_builder.py again with explicit path and log
- Check: `cat ~/model-core/test_venv_builder.log | grep -i error`

### NumPy version errors
- model-setup automatically pins `numpy<2`
- If manually installing packages, avoid upgrading numpy

### Installation Fails with "Unknown install type"
This was a bug in earlier versions. Update to latest:
```bash
git pull origin develop
```

## Jetson-Specific Requirements

- **JetPack**: 6.x (R36.x in `/etc/nv_tegra_release`)
- **Python**: 3.10
- **CUDA**: 12.6 (from JetPack)
- **Special**: PyTorch wheels from NVIDIA's redist, not PyPI

See [GPU/docs/Jetson Orin Nano — PyTorch & TensorFlow Limitations.md](../GPU/docs/Jetson%20Orin%20Nano%20—%20PyTorch%20&%20TensorFlow%20Limitations.md) for detailed compatibility matrix.

## Development

### Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

### Testing

```bash
# Test hardware detection
python -c "from model_setup.hardware_detector import HardwareDetector; \
    h = HardwareDetector().detect(); print(h)"

# Test venv creation with logging
python test_venv_builder.py /tmp/test_venv --config /tmp/test_config.json --log-file /tmp/test.log

# Test with all backends
python test_venv_builder.py /tmp/test_venv --all --config /tmp/test_config.json

# View detailed logs
cat /tmp/test.log | grep -E "(prerequisites|Missing|detected)"
```

## License

MIT License - See LICENSE file

## Related Projects

- [model-core](https://github.com/Alex-Glebov/model-core) - ML training runtime (uses this setup)
- [pivots-api](https://github.com/Alex-Glebov/pivots-api) - Prediction API service
