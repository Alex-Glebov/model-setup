# Model Setup

Automated hardware detection and virtual environment setup for ML training on NVIDIA Jetson, CUDA, ROCm, and CPU-only systems.

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Linux%20aarch64%20%7C%20x86_64-blue)](https://github.com/Alex-Glebov/model-setup)
[![GPU](https://img.shields.io/badge/GPU-Jetson%20%7C%20CUDA%20%7C%20ROCm%20%7C%20CPU-green.svg)]()

## Features

- **Hardware Auto-Detection**: Automatically detects GPU type (Jetson/CUDA/ROCm/CPU)
- **Platform-Specific Setup**: Installs optimal PyTorch version for your hardware
- **Jetson Optimized**: Full support for Jetson Orin with sm_87, cuSPARSELt
- **No Sudo Required**: All dependencies installed locally in venv
- **PyTorch Preferred**: Uses PyTorch over TensorFlow (better GPU support on Jetson)

## Quick Start

```bash
git clone https://github.com/Alex-Glebov/model-setup.git
cd model-setup
python test_venv_builder.py /path/to/venv --config /path/to/.hardware_config.json
```

Example:
```bash
python test_venv_builder.py ~/model-core/venv --config ~/model-core/.hardware_config.json
```

## Hardware Support

| Platform | Status | PyTorch | Notes |
|----------|--------|---------|-------|
| NVIDIA Jetson Orin | ✅ Supported | 2.5.0 (NVIDIA wheel) | sm_87, cuSPARSELt 0.7.1.0 |
| NVIDIA CUDA | ✅ Supported | 2.x (PyPI) | CUDA 12.1 |
| AMD ROCm | ✅ Supported | 2.x (PyPI) | ROCm 5.7 |
| CPU Only | ✅ Supported | 2.x (PyPI) | No GPU acceleration |

### Why PyTorch over TensorFlow?

- **Jetson**: NVIDIA provides official PyTorch wheels for JetPack 6.x; TensorFlow GPU support is not officially stable
- **Performance**: Better sm_87 (Orin) compute capability support
- **Installation**: PyTorch wheels include all dependencies; TensorFlow requires manual dependency management

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

1. **PyTorch** - Hardware-specific version:
   - Jetson: NVIDIA's wheel from `developer.download.nvidia.com`
   - CUDA: `torch` from PyPI with CUDA 12.1
   - ROCm: `torch` from PyPI with ROCm 5.7
   - CPU: `torch` from PyPI CPU-only

2. **cuSPARSELt** (Jetson only):
   - Version 0.7.1.0
   - Installed locally in `venv/lib/cuda/lib/`
   - No system-wide changes (no sudo)

3. **NumPy**:
   - Version < 2.0 (pinned for PyTorch compatibility)
   - Bundled with PyTorch wheel

4. **Other Dependencies**:
   - Read from `model-core/requirements.txt`
   - Skips packages already provided by PyTorch

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
│   ├── hardware_detector.py    # Hardware detection logic
│   └── venv_builder.py         # Venv creation and setup
├── test_venv_builder.py       # CLI entry point
└── README.md                   # This file
```

## Configuration Priority

1. **Explicit** (highest): `~/GPU/GPU_VARIANT.txt` containing `jetson`, `cuda`, or `rocm`
2. **Auto-detect**: PyTorch CUDA availability check
3. **Platform inference**: `aarch64` → likely Jetson

## Troubleshooting

### "libcusparseLt.so.0: cannot open shared object file"
- Venv not activated: Run `source venv/bin/activate`
- Or missing LD_LIBRARY_PATH when using direct Python path

### "No module named 'torch'"
- Venv creation failed - check hardware detection output
- Try running test_venv_builder.py again with explicit path

### NumPy version errors
- model-setup automatically pins `numpy<2`
- If manually installing packages, avoid upgrading numpy

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

# Test venv creation (dry run)
python test_venv_builder.py /tmp/test_venv --config /tmp/test_config.json
```

## License

MIT License - See LICENSE file

## Related Projects

- [model-core](https://github.com/Alex-Glebov/model-core) - ML training runtime (uses this setup)
- [pivots-api](https://github.com/Alex-Glebov/pivots-api) - Prediction API service
