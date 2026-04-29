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
- [[WSL Support]] - Windows Subsystem for Linux
- [[Multi-Backend]] - Installing multiple backends

## What is model-setup?

Model-setup is a Python tool that automatically:
1. Detects your GPU hardware (Jetson, CUDA, ROCm, or CPU) and version
2. Creates a virtual environment with Keras 3.x + your chosen backend(s)
3. Installs all dependencies without requiring sudo
4. Configures library paths for GPU acceleration
5. Supports Linux, Windows+WSL, and multiple backends

## Architecture Overview

### Keras 3.x as Unified Frontend

Model-setup implements **Keras 3.x with multiple execution backends**:

```
Your Model Code (uses keras API)
         │
         ▼
┌─────────────────┐
│   Keras 3.x     │  ← Unified API: layers, models, training
│  (pip install)  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│  torch │ │tensorflow│  ← Execution backends
│Backend │ │ Backend  │     (both can be installed)
└────┬───┘ └────┬─────┘
     │          │
     └────┬─────┘
          ▼
   GPU Hardware
```

**Key Concept**: Write model code once using Keras API, execute on TensorFlow or PyTorch interchangeably. Switch backends by changing one environment variable.

## Quick Start

```bash
# Basic install
python model-setup ~/venv --config ~/config.json

# Install all working backends
python model-setup ~/venv --all --config ~/config.json

# With logging
python model-setup ~/venv --log-file ~/setup.log --config ~/config.json
```

## Project Status

- ✅ Jetson Orin support (sm_87, cuSPARSELt auto-detected)
- ✅ CUDA support (auto-detects 11.8, 12.1, 12.4, 12.8+)
- ✅ ROCm support (auto-detects 5.6, 5.7, 6.0+)
- ✅ CPU-only fallback
- ✅ Windows WSL support (WSL 2 with GPU passthrough)
- ✅ Multi-backend support (torch + tensorflow)
- ✅ Detailed diagnostics (logs WHY GPU failed)
- 🚧 Docker containers (planned)
- 🚧 Multi-GPU support (planned)

## Related Repositories

- [model-core](https://github.com/Alex-Glebov/model-core) - ML training runtime
- [pivots-api](https://github.com/Alex-Glebov/pivots-api) - Prediction API
