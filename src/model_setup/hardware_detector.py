"""Hardware detection for ML training environments.

Supports: Jetson Orin, CUDA GPUs, ROCm GPUs, CPU-only
"""

import json
import logging
import os
import platform
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Hardware information for ML training."""
    platform: str
    machine: str
    gpu_type: Optional[str]  # 'jetson', 'cuda', 'rocm', None
    gpu_name: Optional[str]
    gpu_memory_mb: Optional[int]
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    compute_capability: Optional[str]  # e.g., "8.7" for Orin
    preferred_backend: str  # 'pytorch', 'tensorflow', 'cpu'
    gpu_available: bool

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "HardwareInfo":
        return cls(**data)


class HardwareDetector:
    """Detects hardware capabilities for ML training."""

    def __init__(self):
        self.platform = platform.system()
        self.machine = platform.machine()

    def detect(self) -> HardwareInfo:
        """Detect hardware configuration."""
        logger.info(f"Detecting hardware on {self.platform} {self.machine}")

        # Check for Jetson
        if self._is_jetson():
            return self._detect_jetson()

        # Check for CUDA
        if self._has_cuda():
            return self._detect_cuda()

        # Check for ROCm
        if self._has_rocm():
            return self._detect_rocm()

        # CPU only
        return HardwareInfo(
            platform=self.platform,
            machine=self.machine,
            gpu_type=None,
            gpu_name=None,
            gpu_memory_mb=None,
            cuda_version=None,
            cudnn_version=None,
            compute_capability=None,
            preferred_backend='cpu',
            gpu_available=False
        )

    def _is_jetson(self) -> bool:
        """Check if running on Jetson hardware."""
        return (
            self.machine == 'aarch64' and
            Path('/etc/nv_tegra_release').exists()
        )

    def _has_cuda(self) -> bool:
        """Check if CUDA is available."""
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _has_rocm(self) -> bool:
        """Check if ROCm is available."""
        try:
            result = subprocess.run(
                ['rocm-smi'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _detect_jetson(self) -> HardwareInfo:
        """Detect Jetson-specific hardware."""
        logger.info("Jetson hardware detected")

        # Read JetPack version
        jetpack_version = self._read_jetpack_version()

        # Try to detect GPU via PyTorch
        gpu_name = "Unknown"
        gpu_memory_mb = None
        compute_capability = None

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                cc = torch.cuda.get_device_capability(0)
                compute_capability = f"{cc[0]}.{cc[1]}"
        except ImportError:
            pass

        return HardwareInfo(
            platform=self.platform,
            machine=self.machine,
            gpu_type='jetson',
            gpu_name=gpu_name,
            gpu_memory_mb=gpu_memory_mb,
            cuda_version=None,  # Will detect via nvcc
            cudnn_version=None,  # Will detect via cudnn
            compute_capability=compute_capability,
            preferred_backend='pytorch',  # NVIDIA preference
            gpu_available=compute_capability is not None
        )

    def _detect_cuda(self) -> HardwareInfo:
        """Detect CUDA GPU."""
        logger.info("CUDA GPU detected")

        gpu_name = "Unknown"
        gpu_memory_mb = None
        cuda_version = None

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 2:
                    gpu_name = parts[0].strip()
                    mem_str = parts[1].strip()
                    # Parse "8192 MiB"
                    if 'MiB' in mem_str:
                        gpu_memory_mb = int(mem_str.replace('MiB', '').strip())
        except (subprocess.SubprocessError, FileNotFoundError, ValueError):
            pass

        # Get CUDA version
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'release':
                                cuda_version = parts[i + 1].rstrip(',')
                                break
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        return HardwareInfo(
            platform=self.platform,
            machine=self.machine,
            gpu_type='cuda',
            gpu_name=gpu_name,
            gpu_memory_mb=gpu_memory_mb,
            cuda_version=cuda_version,
            cudnn_version=None,
            compute_capability=None,
            preferred_backend='pytorch',
            gpu_available=True
        )

    def _detect_rocm(self) -> HardwareInfo:
        """Detect ROCm GPU."""
        logger.info("ROCm GPU detected")

        return HardwareInfo(
            platform=self.platform,
            machine=self.machine,
            gpu_type='rocm',
            gpu_name="AMD GPU",
            gpu_memory_mb=None,
            cuda_version=None,
            cudnn_version=None,
            compute_capability=None,
            preferred_backend='pytorch',
            gpu_available=True
        )

    def _read_jetpack_version(self) -> Optional[str]:
        """Read JetPack version from system file."""
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                # Parse version from line like "# R36 (release), REVISION: 5.0"
                if 'REVISION:' in content:
                    parts = content.split('REVISION:')
                    if len(parts) > 1:
                        return parts[1].split(',')[0].strip()
        except (FileNotFoundError, PermissionError, IOError):
            pass
        return None


def detect_and_save(output_path: str) -> HardwareInfo:
    """Detect hardware and save configuration to file.

    Args:
        output_path: Path to write JSON config

    Returns:
        Detected hardware information
    """
    detector = HardwareDetector()
    info = detector.detect()

    # Ensure directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write config
    with open(output_path, 'w') as f:
        json.dump(info.to_dict(), f, indent=2)

    logger.info(f"Hardware config saved to {output_path}")
    return info


if __name__ == '__main__':
    # CLI usage
    import sys

    logging.basicConfig(level=logging.INFO)

    output_file = sys.argv[1] if len(sys.argv) > 1 else 'hardware_config.json'
    info = detect_and_save(output_file)

    print(f"\nDetected Hardware:")
    print(f"  Platform: {info.platform} {info.machine}")
    print(f"  GPU Type: {info.gpu_type or 'None'}")
    print(f"  GPU Name: {info.gpu_name or 'N/A'}")
    print(f"  Preferred Backend: {info.preferred_backend}")
    print(f"  GPU Available: {info.gpu_available}")
