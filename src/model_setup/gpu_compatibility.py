"""GPU compatibility testing for PyTorch.

Tests if PyTorch GPU installation actually works before committing.
Catches issues like:
- ROCm missing kernels for specific GPU architectures
- CUDA out of memory
- Driver mismatches
"""

import logging
import platform
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_python_path(venv_path: Path) -> Path:
    """Get Python executable path for venv (platform-aware).

    Args:
        venv_path: Path to venv

    Returns:
        Path to Python executable
    """
    if platform.system() == 'Windows':
        return venv_path / 'Scripts' / 'python.exe'
    return venv_path / 'bin' / 'python'


def test_gpu_compatibility(venv_path: Path, timeout: int = 60) -> tuple[bool, str]:
    """Test if PyTorch GPU is actually functional.

    Runs in the target venv to test the installed PyTorch.
    Tests tensor operations, LSTM (the operation that failed on ROCm iGPU),
    and backward pass.

    Args:
        venv_path: Path to venv with PyTorch installed
        timeout: Maximum seconds to wait for test

    Returns:
        (success, message) tuple
    """
    python_path = _get_python_path(venv_path)

    if not python_path.exists():
        return False, f"Python not found at {python_path}"

    test_script = '''
import torch
import sys

try:
    # Test 1: CUDA available
    if not torch.cuda.is_available():
        print("CUDA_NOT_AVAILABLE")
        sys.exit(1)

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    print(f"DEVICE:{device_name}")

    # Test 2: Basic tensor operations
    x = torch.randn(100, 100, device=device)
    y = x @ x.T
    z = y.sum()

    # Test 3: LSTM (the operation that failed on ROCm iGPU)
    # This is the critical test - ROCm may fail here with "invalid device function"
    lstm = torch.nn.LSTM(10, 20, num_layers=2, batch_first=True).to(device)
    input_t = torch.randn(8, 10, 10, device=device)
    output, (hn, cn) = lstm(input_t)

    # Test 4: Backward pass
    loss = output.mean()
    loss.backward()

    # Test 5: Multihead attention (if available)
    try:
        attn = torch.nn.MultiheadAttention(20, 4, batch_first=True).to(device)
        attn_out, _ = attn(output, output, output)
        print("ATTN_OK")
    except Exception as e:
        print(f"ATTN_SKIP:{e}")

    print("GPU_OK")
    sys.exit(0)

except RuntimeError as e:
    error_msg = str(e)
    if "invalid device function" in error_msg:
        print(f"GPU_INCOMPATIBLE:{error_msg}")
    elif "out of memory" in error_msg:
        print(f"GPU_OOM:{error_msg}")
    else:
        print(f"GPU_RUNTIME_ERROR:{error_msg}")
    sys.exit(1)

except Exception as e:
    print(f"GPU_ERROR:{type(e).__name__}: {e}")
    sys.exit(1)
'''

    try:
        result = subprocess.run(
            [str(python_path), '-c', test_script],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output = result.stdout.strip()
        errors = result.stderr.strip()

        if result.returncode == 0 and "GPU_OK" in output:
            # Extract device name from output
            device_name = "Unknown"
            for line in output.split('\n'):
                if line.startswith("DEVICE:"):
                    device_name = line.split(":", 1)[1]
                    break
            return True, f"GPU OK: {device_name}"

        # Parse failure
        if "GPU_INCOMPATIBLE" in output:
            return False, "GPU kernel incompatible (e.g., ROCm iGPU)"
        elif "GPU_OOM" in output:
            return False, "GPU out of memory"
        elif "CUDA_NOT_AVAILABLE" in output:
            return False, "CUDA not available after install"
        elif "GPU_RUNTIME_ERROR" in output:
            error = output.split(":", 1)[1] if ":" in output else output
            return False, f"GPU runtime error: {error}"
        else:
            return False, f"Test failed: {output} {errors}"

    except subprocess.TimeoutExpired:
        return False, f"GPU test timed out after {timeout}s"
    except Exception as e:
        return False, f"Test error: {e}"


def test_cpu_compatibility(venv_path: Path, timeout: int = 30) -> tuple[bool, str]:
    """Test if PyTorch CPU is functional.

    Args:
        venv_path: Path to venv with PyTorch installed
        timeout: Maximum seconds to wait for test

    Returns:
        (success, message) tuple
    """
    python_path = _get_python_path(venv_path)

    test_script = '''
import torch
import sys

try:
    # Test CPU operations
    x = torch.randn(10, 10)
    y = x @ x.T

    # Verify no GPU detected (for CPU-only install)
    has_cuda = torch.cuda.is_available()

    print(f"CPU_OK:CUDA_AVAILABLE={has_cuda}")
    sys.exit(0)

except Exception as e:
    print(f"CPU_ERROR:{e}")
    sys.exit(1)
'''

    try:
        result = subprocess.run(
            [str(python_path), '-c', test_script],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0 and "CPU_OK" in result.stdout:
            cuda_status = result.stdout.strip().split("=")[-1]
            return True, f"CPU OK (CUDA available={cuda_status})"
        else:
            return False, f"CPU test failed: {result.stdout} {result.stderr}"

    except Exception as e:
        return False, f"CPU test error: {e}"


def quick_gpu_check() -> dict:
    """Quick check without full test - for build queue decisions.

    Returns info about available GPU without importing PyTorch.
    Uses system commands where possible.
    """
    info = {
        'has_nvidia_gpu': False,
        'has_amd_gpu': False,
        'driver_issues': [],
    }

    # Check for NVIDIA
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
        if result.returncode == 0:
            info['has_nvidia_gpu'] = True
    except:
        pass

    # Check for AMD (ROCm)
    try:
        result = subprocess.run(['rocminfo'], capture_output=True, timeout=5)
        if result.returncode == 0 and b'gfx' in result.stdout:
            info['has_amd_gpu'] = True
    except:
        pass

    return info
