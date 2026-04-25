# WSL Support (Windows Subsystem for Linux)

Model-setup fully supports WSL 2 for GPU passthrough on Windows. This allows you to use your NVIDIA GPU within a Linux environment on Windows.

## Prerequisites

### Required

- **Windows 10 version 2004+** or **Windows 11**
- **WSL 2** (WSL 1 does not support GPU passthrough)
- **NVIDIA GPU drivers** installed on Windows host

### Checking WSL Version

```powershell
# In PowerShell or Command Prompt
wsl --version
# Expected: WSL version: 2.x.x
```

### Checking WSL 2

```bash
# In WSL terminal
cat /proc/version
# Should contain: "microsoft" or "WSL"
```

## Installation Steps

### Step 1: Install/Update WSL 2

```powershell
# Install WSL (if not already installed)
wsl --install

# Or update existing WSL
wsl --update

# Set default version to WSL 2
wsl --set-default-version 2
```

### Step 2: Install NVIDIA Drivers on Windows

**Important**: Install on Windows (the host), NOT in WSL.

1. Download from: https://www.nvidia.com/drivers/
2. Select your GPU and Windows version
3. Install and reboot Windows

### Step 3: Verify GPU in WSL

```bash
# In WSL terminal
ls /usr/lib/wsl/lib/

# Should see files like:
# libcuda.so.1
# libd3d12.so
# libdxcore.so
```

If these files are missing, the GPU is not properly passed through.

### Step 4: Install model-setup

```bash
# Clone repository
git clone https://github.com/Alex-Glebov/model-setup.git
cd model-setup

# Run setup with logging
python test_venv_builder.py ~/venv \
    --config ~/hardware_config.json \
    --log-file ~/wsl-setup.log
```

## What model-setup Checks

When running in WSL, model-setup automatically:

1. **Detects WSL environment**
   ```
   INFO - WSL environment detected
   ```

2. **Checks WSL version**
   ```
   INFO - WSL 2 confirmed
   # or
   WARNING - WSL 1 detected - WSL 2 required for GPU support
   ```

3. **Verifies GPU passthrough**
   ```
   WARNING - WSL GPU support not detected (/usr/lib/wsl/lib missing)
   ```

4. **Checks DirectX support**
   ```
   WARNING - WSL DirectX support missing (libdxcore.so not found)
   ```

## Troubleshooting

### "WSL 1 detected - WSL 2 required for GPU support"

**Problem**: Running WSL 1 instead of WSL 2.

**Solution**:
```powershell
# Convert your distribution to WSL 2
wsl --set-version Ubuntu 2

# Or set default for new distributions
wsl --set-default-version 2
```

### "/usr/lib/wsl/lib missing"

**Problem**: GPU drivers not properly passed through.

**Solution**:
1. Ensure NVIDIA drivers are installed on Windows (not in WSL)
2. Update WSL: `wsl --update`
3. Restart WSL: `wsl --shutdown` then reopen
4. Verify again: `ls /usr/lib/wsl/lib/`

### CUDA not available in PyTorch

**Problem**: PyTorch installed but can't see GPU.

**Check**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

**Solution**:
- Make sure you're using PyTorch with CUDA support (model-setup handles this)
- Check Windows NVIDIA drivers are installed
- Verify `/usr/lib/wsl/lib/libcuda.so.1` exists

## Performance Considerations

### WSL 2 vs Native Linux

- **WSL 2** uses a lightweight VM with full Linux kernel
- **GPU performance**: Near-native performance for compute (CUDA)
- **File I/O**: Slightly slower for cross-filesystem operations
- **Recommendation**: Keep project files in WSL filesystem (`/home/user/...`) not Windows (`/mnt/c/...`)

### Best Practices

1. **Store projects in WSL**:
   ```bash
   # Good: Fast I/O
   cd ~/projects

   # Slower: Cross-filesystem access
   cd /mnt/c/Users/Name/projects
   ```

2. **Use WSL terminal**:
   - Windows Terminal with WSL profile
   - VS Code with WSL extension
   - Not: PowerShell/CMD for Python execution

3. **Update WSL regularly**:
   ```powershell
   wsl --update
   ```

## How It Works

### GPU Passthrough Architecture

```
Windows Host
├── NVIDIA Driver (installed on Windows)
├── WSL 2 VM
│   └── Linux Kernel
│       └── /usr/lib/wsl/lib/ (GPU libraries mounted)
│           ├── libcuda.so (CUDA driver)
│           ├── libd3d12.so (DirectX)
│           └── libdxcore.so (DirectML)
```

### Detection Flow

1. model-setup detects WSL via `/proc/version` or `WSL_DISTRO_NAME`
2. Checks WSL version (must be 2 for GPU)
3. Verifies GPU libraries are mounted at `/usr/lib/wsl/lib`
4. Proceeds with CUDA installation as normal

## Known Limitations

- **ROCm on WSL**: Not officially supported by AMD
- **GUI applications**: May require additional X server setup
- **Docker**: Use Docker Desktop WSL2 backend for GPU support

## References

- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [Microsoft WSL Documentation](https://docs.microsoft.com/windows/wsl/)
