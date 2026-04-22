# API Reference

Complete API documentation for model-setup modules.

## Table of Contents

- [HardwareDetector](#hardwaredetector)
- [HardwareInfo](#hardwareinfo)
- [VenvBuilder](#venvbuilder)
- [create_venv_for_hardware](#create_venv_for_hardware)

## HardwareDetector

```python
from model_setup.hardware_detector import HardwareDetector
```

Detects GPU hardware and capabilities.

### Methods

#### `detect() -> HardwareInfo`

Detects hardware and returns hardware information.

**Returns**: `HardwareInfo` object with gpu_type, gpu_name, compute_capability, etc.

**Example**:
```python
detector = HardwareDetector()
info = detector.detect()
print(info.gpu_type)  # 'jetson', 'cuda', 'rocm', or None
```

#### `detect_jetson() -> Optional[str]`

Specifically detects Jetson hardware.

**Returns**: Jetson device name or None

**Detection method**:
- Checks `/etc/nv_tegra_release` for R36/R35
- Checks `/proc/device-tree/model` for Jetson name

#### `detect_cuda() -> bool`

Detects NVIDIA CUDA GPU availability.

**Returns**: True if CUDA-capable GPU found

**Note**: Requires PyTorch to be installed to check

#### `detect_rocm() -> bool`

Detects AMD ROCm GPU availability.

**Returns**: True if ROCm GPU found

---

## HardwareInfo

```python
from model_setup.hardware_detector import HardwareInfo
```

Data class containing hardware information.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `gpu_type` | Optional[str] | 'jetson', 'cuda', 'rocm', or None |
| `gpu_name` | Optional[str] | GPU model name |
| `compute_capability` | Optional[str] | CUDA compute capability (e.g., '8.7') |
| `cuda_version` | Optional[str] | CUDA version if available |
| `cudnn_version` | Optional[str] | cuDNN version if available |

### Methods

#### `to_dict() -> dict`

Converts to dictionary for JSON serialization.

**Example**:
```python
info = detector.detect()
config = info.to_dict()
# {
#   "gpu_type": "jetson",
#   "gpu_name": "Orin",
#   "compute_capability": "8.7",
#   "cuda_version": "12.6",
#   "cudnn_version": "9.0"
# }
```

---

## VenvBuilder

```python
from model_setup.venv_builder import VenvBuilder
```

Creates and configures virtual environments for ML training.

### Constructor

#### `VenvBuilder(venv_path: str, hardware_info: Optional[HardwareInfo] = None)`

**Parameters**:
- `venv_path`: Path where venv will be created
- `hardware_info`: Hardware information (optional, detected if not provided)

**Example**:
```python
from model_setup.hardware_detector import HardwareDetector
from model_setup.venv_builder import VenvBuilder

info = HardwareDetector().detect()
builder = VenvBuilder("~/myvenv", info)
```

### Methods

#### `create() -> Path`

Creates the virtual environment with hardware-specific configuration.

**Returns**: Path to created venv

**Example**:
```python
venv_path = builder.create()
print(f"Created: {venv_path}")
```

**Process**:
1. Creates venv using `python -m venv`
2. Calls `_apply_hardware_config()` if hardware_info provided
3. Returns venv path

#### `_apply_hardware_config()`

Applies hardware-specific configuration.

**Called automatically by `create()`**

Dispatches to:
- `_configure_jetson()` for Jetson devices
- `_configure_cuda()` for NVIDIA CUDA GPUs
- `_configure_rocm()` for AMD ROCm GPUs
- `_configure_cpu()` for CPU-only

#### `_configure_jetson()`

Configures venv for Jetson devices.

**Steps**:
1. Creates `venv/lib/cuda/lib/` directory
2. Downloads and installs cuSPARSELt 0.7.1.0
3. Patches activate script for LD_LIBRARY_PATH
4. Installs PyTorch 2.5.0 from NVIDIA wheel
5. Installs remaining dependencies from requirements.txt

**PyTorch Wheel**:
```
https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/
torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

#### `_configure_cuda()`

Configures venv for CUDA GPUs.

**Installs**:
- PyTorch with CUDA 12.1 from PyPI
- Remaining dependencies from requirements.txt

#### `_configure_rocm()`

Configures venv for ROCm GPUs.

**Installs**:
- PyTorch with ROCm 5.7 from PyPI
- Remaining dependencies from requirements.txt

#### `_configure_cpu()`

Configures venv for CPU-only.

**Installs**:
- PyTorch CPU version from PyPI
- Remaining dependencies from requirements.txt

---

## create_venv_for_hardware

```python
from model_setup.venv_builder import create_venv_for_hardware
```

Convenience function for one-step venv creation.

### Signature

```python
def create_venv_for_hardware(
    venv_path: str,
    output_config_path: Optional[str] = None
) -> tuple[Path, HardwareInfo]:
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `venv_path` | str | Path to create venv |
| `output_config_path` | Optional[str] | Path to write hardware config JSON |

### Returns

Tuple of `(venv_path, hardware_info)`

### Example

```python
from model_setup.venv_builder import create_venv_for_hardware

venv, hardware = create_venv_for_hardware(
    "/home/user/model-core/venv",
    "/home/user/model-core/.hardware_config.json"
)

print(f"Venv: {venv}")
print(f"GPU: {hardware.gpu_name}")
```

### CLI Usage

```bash
python -m model_setup.venv_builder ~/venv --config ~/config.json
```

Or using the test script:

```bash
python test_venv_builder.py ~/venv --config ~/config.json
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""Create ML environment for detected hardware."""

import logging
from model_setup.venv_builder import create_venv_for_hardware

logging.basicConfig(level=logging.INFO)

# Create venv
venv, hardware = create_venv_for_hardware(
    "/home/user/model-core/venv",
    "/home/user/model-core/.hardware_config.json"
)

print(f"\n✓ Virtual environment: {venv}")
print(f"  Hardware: {hardware.gpu_type or 'CPU'}")
print(f"  GPU: {hardware.gpu_name or 'N/A'}")
print(f"\nTo activate:")
print(f"  source {venv}/bin/activate")
```

## Error Handling

All methods raise standard Python exceptions:

- `RuntimeError`: Unsupported JetPack version, missing dependencies
- `subprocess.CalledProcessError`: Installation command failure
- `FileNotFoundError`: Missing required files

Example:
```python
try:
    venv = builder.create()
except RuntimeError as e:
    print(f"Setup failed: {e}")
```
