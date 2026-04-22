# Model Setup Wiki

Welcome to the model-setup wiki! This is the documentation hub for automated ML environment setup.

## Quick Navigation

- [[Installation Guide]] - Step-by-step setup instructions
- [[Hardware Compatibility]] - Supported GPUs and platforms
- [[Troubleshooting]] - Common issues and solutions
- [[API Reference]] - Module documentation
- [[Jetson Setup]] - Jetson-specific instructions
- [[CUDA Setup]] - NVIDIA CUDA setup
- [[ROCm Setup]] - AMD ROCm setup

## What is model-setup?

Model-setup is a Python tool that automatically:
1. Detects your GPU hardware (Jetson, CUDA, ROCm, or CPU)
2. Creates a virtual environment with the correct PyTorch version
3. Installs all dependencies without requiring sudo
4. Configures library paths for GPU acceleration

## Quick Start

```bash
python test_venv_builder.py ~/venv --config ~/config.json
```

## Project Status

- ✅ Jetson Orin support (sm_87, cuSPARSELt)
- ✅ CUDA support (CUDA 12.1)
- ✅ ROCm support (ROCm 5.7)
- ✅ CPU-only fallback
- 🚧 Docker containers (planned)
- 🚧 Multi-GPU support (planned)

## Related Repositories

- [model-core](https://github.com/Alex-Glebov/model-core) - ML training runtime
- [pivots-api](https://github.com/Alex-Glebov/pivots-api) - Prediction API
