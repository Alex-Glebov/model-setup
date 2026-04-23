"""Virtual environment builder for ML training.

Creates venv with hardware-specific configuration and test-before-commit.
Archives failed attempts for debugging.
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .hardware_detector import HardwareDetector, HardwareInfo
from .gpu_compatibility import test_gpu_compatibility, test_cpu_compatibility
from .pip_version_checker import can_install_backend

logger = logging.getLogger(__name__)


class VenvBuilder:
    """Builds virtual environment with hardware-specific configuration.

    Implements test-before-commit strategy:
    1. Install GPU variant to destination
    2. Test immediately
    3. If pass - keep, install remaining deps
    4. If fail - archive with logs, try next variant
    5. CPU is guaranteed fallback
    """

    def __init__(
        self,
        venv_path: str,
        hardware_info: Optional[HardwareInfo] = None,
        on_fail: str = 'n'  # 'a'=auto-delete, 'y'=ask, 'n'=keep
    ):
        self.venv_path = Path(venv_path)
        self.hardware_info = hardware_info
        self.on_fail = on_fail
        self._pip_path: Optional[Path] = None
        self._log_file: Optional[Path] = None
        self._file_handler: Optional[logging.FileHandler] = None

    @property
    def pip_path(self) -> Path:
        """Get path to venv's pip executable."""
        if self._pip_path is None:
            self._pip_path = self.venv_path / 'bin' / 'pip'
        return self._pip_path

    def create(self) -> Path:
        """Create virtual environment with test-before-commit.

        Returns:
            Path to created venv
        """
        logger.info("=" * 60)
        logger.info("Venv Builder - Test Before Commit")
        logger.info("=" * 60)

        # Preserve existing venv as venv.orig if it exists
        if self.venv_path.exists():
            orig_path = self._rename_to_orig()
            logger.info(f"Renamed existing venv to {orig_path}")

        # Build installation priority queue
        install_queue = self._build_install_queue()
        logger.info(f"Installation queue: {install_queue}")

        # Try each variant until one passes
        for install_type in install_queue:
            logger.info(f"\nTrying {install_type.upper()}...")

            # Archive any existing venv from previous attempt
            if self.venv_path.exists():
                self._archive_venv(install_type, failed=True)

            # Create fresh venv with logging
            self._create_venv(self.venv_path)

            # Install PyTorch variant
            try:
                self._install_pytorch(install_type)
            except Exception as e:
                logger.error(f"Install failed: {e}")
                self._archive_venv(f"{install_type}-install-failed")
                continue

            # TEST: Verify GPU actually works
            logger.info(f"Testing {install_type}...")
            success, msg = self._test_installation(install_type)

            if success:
                logger.info(f"✓ {install_type} PASSED: {msg}")
                self._install_remaining_deps()
                self._cleanup_logging()

                # Optionally archive successful attempts
                # self._cleanup_archives(keep_last=1)

                return self.venv_path
            else:
                logger.warning(f"✗ {install_type} FAILED: {msg}")
                archive_path = self._archive_venv(f"{install_type}-failed")
                self._handle_failure(archive_path, install_type)

        # Should never reach here (CPU always works)
        raise RuntimeError("All installation options failed including CPU")

    def _build_install_queue(self) -> list[str]:
        """Build installation priority queue based on detected hardware.

        Filters by PyPI availability - only includes backends that can be installed.
        """
        queue = []

        if not self.hardware_info:
            # Check if CPU backend is available on PyPI
            can_install, _ = can_install_backend('torch')
            if can_install:
                queue.append('cpu')
            return queue

        gpu_type = self.hardware_info.gpu_type

        # Map hardware to preferred Keras 3.x backend
        # Priority: GPU backends first
        if gpu_type == 'jetson':
            # Jetson works best with torch backend
            can_install, _ = can_install_backend('torch')
            if can_install:
                queue.append('jetson')

        elif gpu_type == 'cuda':
            # CUDA works with torch or tensorflow
            for backend in ['torch', 'tensorflow']:
                can_install, _ = can_install_backend(backend)
                if can_install:
                    queue.append(backend)

        elif gpu_type == 'rocm':
            # ROCm works best with torch
            can_install, _ = can_install_backend('torch')
            if can_install:
                queue.append('rocm')

        # CPU fallback using torch (lightest weight)
        can_install, _ = can_install_backend('torch')
        if can_install:
            queue.append('cpu')

        return queue

    def _create_venv(self, venv_path: Path):
        """Create venv and setup logging."""
        logger.info(f"Creating venv at {venv_path}")

        # Create venv
        subprocess.run(
            [sys.executable, '-m', 'venv', str(venv_path)],
            check=True
        )

        # Setup per-venv logging
        self._setup_venv_logging(venv_path)

        # Log system info
        self._log_system_info()

    def _setup_venv_logging(self, venv_path: Path):
        """Setup logging to file inside venv."""
        # Cleanup previous handler
        self._cleanup_logging()

        # Create log file in venv
        self._log_file = venv_path / 'install.log'
        self._file_handler = logging.FileHandler(self._log_file)
        self._file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(self._file_handler)

        logger.info("=" * 60)
        logger.info("Installation Log Started")
        logger.info("=" * 60)

    def _cleanup_logging(self):
        """Remove file handlers."""
        if self._file_handler:
            self._file_handler.close()
            logger.removeHandler(self._file_handler)
            self._file_handler = None
            self._log_file = None

    def _log_system_info(self):
        """Log system information for debugging."""
        logger.info(f"Platform: {sys.platform} {platform.machine()}")
        logger.info(f"Python: {sys.version}")

        if self.hardware_info:
            logger.info(f"Detected GPU Type: {self.hardware_info.gpu_type}")
            logger.info(f"GPU Name: {self.hardware_info.gpu_name}")
            logger.info(f"Compute Capability: {self.hardware_info.compute_capability}")

        try:
            result = subprocess.run(['nvidia-smi', '-L'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"NVIDIA GPUs:\n{result.stdout}")
        except:
            pass

        try:
            result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Extract GPU name from rocminfo
                for line in result.stdout.split('\n'):
                    if 'Marketing Name' in line and 'Radeon' in line:
                        logger.info(f"AMD GPU: {line.strip()}")
                        break
        except:
            pass

    def _rename_to_orig(self) -> Path:
        """Rename existing venv to venv.orig.

        Returns:
            Path to renamed venv
        """
        # Close any logging first
        self._cleanup_logging()

        # Find unique orig path
        orig_path = self.venv_path.parent / f"{self.venv_path.name}.orig"
        counter = 1
        while orig_path.exists():
            orig_path = self.venv_path.parent / f"{self.venv_path.name}.orig.{counter}"
            counter += 1

        # Rename venv
        shutil.move(str(self.venv_path), str(orig_path))
        logger.info(f"Renamed existing venv to {orig_path}")

        return orig_path

    def _archive_venv(self, suffix: str, failed: bool = False) -> Path:
        """Archive venv by renaming it.

        Args:
            suffix: Name suffix (e.g., 'cuda-failed', 'rocm')
            failed: If True, this is a failed attempt

        Returns:
            Path to archived venv
        """
        # Close logging before moving
        self._cleanup_logging()

        # Find unique archive path
        archive_path = self.venv_path.parent / f"{self.venv_path.name}.{suffix}"
        counter = 1
        while archive_path.exists():
            archive_path = self.venv_path.parent / f"{self.venv_path.name}.{suffix}.{counter}"
            counter += 1

        # Move venv (includes install.log)
        shutil.move(str(self.venv_path), str(archive_path))

        status = "FAILED" if failed else "archived"
        logger.info(f"Venv {status}: {archive_path}")

        return archive_path

    def _handle_failure(self, archive_path: Path, install_type: str):
        """Handle failed installation based on --on-fail setting."""
        if self.on_fail == 'a':
            logger.info("Auto-deleting failed venv (--on-fail=a)")
            shutil.rmtree(archive_path)

        elif self.on_fail == 'y':
            print(f"\nFailed {install_type} venv archived to: {archive_path}")
            print(f"Log file: {archive_path}/install.log")
            response = input(f"Delete {archive_path}? [y/N]: ")
            if response.lower() == 'y':
                shutil.rmtree(archive_path)
                logger.info(f"Deleted {archive_path}")
            else:
                logger.info(f"Kept {archive_path} for inspection")

        else:  # 'n' = keep (default)
            logger.info(f"Kept failed venv for inspection: {archive_path}")
            logger.info(f"Check log: {archive_path}/install.log")

    def _test_installation(self, install_type: str) -> tuple[bool, str]:
        """Test if installed PyTorch variant works."""
        if install_type == 'cpu':
            return test_cpu_compatibility(self.venv_path)
        else:
            return test_gpu_compatibility(self.venv_path)

    def _install_pytorch(self, install_type: str):
        """Install PyTorch variant."""
        logger.info(f"Installing PyTorch ({install_type})...")

        if install_type == 'jetson':
            self._install_pytorch_jetson()
        elif install_type == 'cuda':
            self._install_pytorch_cuda()
        elif install_type == 'rocm':
            self._install_pytorch_rocm()
        elif install_type == 'cpu':
            self._install_pytorch_cpu()
        else:
            raise ValueError(f"Unknown install type: {install_type}")

    def _install_pytorch_jetson(self):
        """Install PyTorch + Keras for Jetson."""
        jetpack_version = self._get_jetpack_version()
        logger.info(f"Detected JetPack: {jetpack_version}")

        if jetpack_version and jetpack_version.startswith('6'):
            pytorch_wheel = (
                "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/"
                "torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
            )
        else:
            raise RuntimeError(f"Unsupported JetPack: {jetpack_version}")

        subprocess.run(
            [str(self.pip_path), 'install', pytorch_wheel],
            check=True
        )
        logger.info("PyTorch (Jetson) installed")

        # Install Keras 3.x
        subprocess.run(
            [str(self.pip_path), 'install', 'keras>=3.0.0'],
            check=True
        )
        logger.info("Keras installed")

    def _install_pytorch_cuda(self):
        """Install PyTorch + Keras for CUDA."""
        subprocess.run(
            [str(self.pip_path), 'install', 'torch', 'torchvision',
             '--index-url', 'https://download.pytorch.org/whl/cu121'],
            check=True
        )
        logger.info("PyTorch (CUDA) installed")

        # Install Keras 3.x
        subprocess.run(
            [str(self.pip_path), 'install', 'keras>=3.0.0'],
            check=True
        )
        logger.info("Keras installed")

    def _install_pytorch_rocm(self):
        """Install PyTorch + Keras for ROCm."""
        subprocess.run(
            [str(self.pip_path), 'install', 'torch', 'torchvision',
             '--index-url', 'https://download.pytorch.org/whl/rocm5.7'],
            check=True
        )
        logger.info("PyTorch (ROCm) installed")

        # Install Keras 3.x
        subprocess.run(
            [str(self.pip_path), 'install', 'keras>=3.0.0'],
            check=True
        )
        logger.info("Keras installed")

    def _install_pytorch_cpu(self):
        """Install PyTorch + Keras for CPU."""
        subprocess.run(
            [str(self.pip_path), 'install', 'torch', 'torchvision',
             '--index-url', 'https://download.pytorch.org/whl/cpu'],
            check=True
        )
        logger.info("PyTorch (CPU) installed")

        # Install Keras 3.x
        subprocess.run(
            [str(self.pip_path), 'install', 'keras>=3.0.0'],
            check=True
        )
        logger.info("Keras installed")

    def _install_remaining_deps(self):
        """Install remaining packages from requirements.txt."""
        logger.info("Installing remaining dependencies...")

        req_file = Path(__file__).parent.parent.parent.parent / 'model-core' / 'requirements.txt'

        if not req_file.exists():
            logger.warning(f"Requirements not found: {req_file}")
            return

        # Skip packages bundled with PyTorch
        pytorch_packages = {'sympy', 'jinja2', 'networkx', 'fsspec',
                           'filelock', 'typing-extensions', 'mpmath', 'MarkupSafe'}

        with open(req_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            pkg_name = line.split('>=')[0].split('==')[0].split('<')[0].lower().replace('_', '-')

            if pkg_name == 'numpy':
                # Pin numpy<2 for PyTorch compatibility
                logger.info("Pinning numpy<2")
                subprocess.run(
                    [str(self.pip_path), 'install', 'numpy<2'],
                    check=False
                )
            elif pkg_name not in pytorch_packages:
                logger.info(f"Installing {line}")
                subprocess.run(
                    [str(self.pip_path), 'install', line],
                    check=False
                )

        logger.info("Dependencies installed")

    def _get_jetpack_version(self) -> Optional[str]:
        """Get JetPack version from system."""
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                if 'R36' in content:
                    return '6.0'
                elif 'R35' in content:
                    return '5.0'
        except (FileNotFoundError, PermissionError, IOError):
            pass
        return None

    def _configure_cuda(self):
        """Configure for CUDA."""
        self._install_pytorch('cuda')
        self._install_remaining_deps()

    def _configure_rocm(self):
        """Configure for ROCm."""
        self._install_pytorch('rocm')
        self._install_remaining_deps()

    def _configure_cpu(self):
        """Configure for CPU."""
        self._install_pytorch('cpu')
        self._install_remaining_deps()


def create_venv_for_hardware(
    venv_path: str,
    output_config_path: Optional[str] = None,
    on_fail: str = 'n'
) -> tuple[Path, HardwareInfo, str]:
    """Create venv configured for detected hardware.

    Args:
        venv_path: Where to create venv
        output_config_path: Where to write hardware config JSON (optional)
        on_fail: Action on failed install - 'a'=auto-delete, 'y'=ask, 'n'=keep

    Returns:
        (venv_path, hardware_info, keras_backend)
    """
    # Detect hardware
    detector = HardwareDetector()
    hardware_info = detector.detect()

    # Create venv with test-before-commit
    builder = VenvBuilder(venv_path, hardware_info, on_fail=on_fail)
    venv = builder.create()

    # Determine keras_backend from hardware_info
    # Map hardware detector's preferred_backend to Keras 3.x backend names
    keras_backend_map = {
        'pytorch': 'torch',
        'tensorflow': 'tensorflow',
        'cpu': 'torch',  # Default to torch for CPU
    }
    keras_backend = keras_backend_map.get(hardware_info.preferred_backend, 'torch')

    # Write hardware config with keras_backend
    if output_config_path:
        import json
        config_path = Path(output_config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = hardware_info.to_dict()
        config['keras_backend'] = keras_backend

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Hardware config written to {config_path}")
        logger.info(f"Keras backend: {keras_backend}")

    # Generate keras_backend.py for model-core
    _generate_keras_backend_py(venv_path, keras_backend)

    return venv, hardware_info, keras_backend


def _generate_keras_backend_py(venv_path: Path, keras_backend: str):
    """Generate keras_backend.py for model-core package.

    This file sets KERAS_BACKEND environment variable before importing keras.
    Other modules import keras from here instead of directly.

    Args:
        venv_path: Path to venv (contains model-core checkout)
        keras_backend: Keras backend name from config['keras_backend']
    """
    # Find model_core directory in the venv's parent (assumes model-core is checked out there)
    model_core_parent = venv_path.parent
    keras_backend_path = model_core_parent / 'model_core' / 'keras_backend.py'

    content = f'''\"\"\"Keras backend initialization.

Auto-generated by model-setup. Do not edit manually.
Backend: {keras_backend}
\"\"\"

import os

# Set Keras backend BEFORE importing keras
os.environ["KERAS_BACKEND"] = "{keras_backend}"

# Import keras with configured backend
import keras

__all__ = ["keras"]
'''

    try:
        keras_backend_path.parent.mkdir(parents=True, exist_ok=True)
        with open(keras_backend_path, 'w') as f:
            f.write(content)
        logger.info(f"Generated {keras_backend_path}")
    except Exception as e:
        logger.warning(f"Could not generate keras_backend.py: {e}")


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Create ML venv for detected hardware')
    parser.add_argument('venv_path', help='Path to create venv')
    parser.add_argument('--config', help='Path to write hardware config JSON')
    parser.add_argument('--on-fail', choices=['a', 'y', 'n'], default='n',
                        help='Action on failed install: a=auto-delete, y=ask, n=keep (default)')
    args = parser.parse_args()

    venv, hardware, keras_backend = create_venv_for_hardware(
        args.venv_path, args.config, args.on_fail
    )

    print(f"\n✓ Virtual environment created at: {venv}")
    print(f"  Hardware: {hardware.gpu_type or 'CPU-only'}")
    print(f"  GPU: {hardware.gpu_name or 'N/A'}")
    print(f"  Keras Backend: {keras_backend}")
    print(f"\nTo activate:")
    print(f"  source {venv}/bin/activate")
