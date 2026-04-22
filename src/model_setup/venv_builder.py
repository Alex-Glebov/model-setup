"""Virtual environment builder for ML training.

Creates venv with hardware-specific configuration.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .hardware_detector import HardwareDetector, HardwareInfo

logger = logging.getLogger(__name__)


class VenvBuilder:
    """Builds virtual environment with hardware-specific configuration."""

    def __init__(self, venv_path: str, hardware_info: Optional[HardwareInfo] = None):
        self.venv_path = Path(venv_path)
        self.hardware_info = hardware_info
        self._pip_path: Optional[Path] = None

    @property
    def pip_path(self) -> Path:
        """Get path to venv's pip executable."""
        if self._pip_path is None:
            self._pip_path = self.venv_path / 'bin' / 'pip'
        return self._pip_path

    def create(self) -> Path:
        """Create virtual environment.

        Returns:
            Path to created venv
        """
        logger.info(f"Creating virtual environment at {self.venv_path}")

        # Create venv using Python's built-in venv
        subprocess.run(
            [sys.executable, '-m', 'venv', str(self.venv_path)],
            check=True
        )

        # Apply hardware-specific configurations
        if self.hardware_info:
            self._apply_hardware_config()

        logger.info(f"Virtual environment created at {self.venv_path}")
        return self.venv_path

    def _apply_hardware_config(self):
        """Apply hardware-specific configurations to venv."""
        if not self.hardware_info:
            return

        gpu_type = self.hardware_info.gpu_type

        if gpu_type == 'jetson':
            self._configure_jetson()
        elif gpu_type == 'cuda':
            self._configure_cuda()
        elif gpu_type == 'rocm':
            self._configure_rocm()
        else:
            self._configure_cpu()

    def _configure_jetson(self):
        """Configure venv for Jetson (Orin, Xavier, etc.).

        Jetson-specific setup:
        - Install cuSPARSELt locally (needed for PyTorch 2.4+)
        - Install PyTorch (matching JetPack version) - BEST GPU support
        - Modify activate script to include LD_LIBRARY_PATH
        """
        logger.info("Configuring for Jetson")

        # Create local CUDA lib directory
        cuda_lib = self.venv_path / 'lib' / 'cuda' / 'lib'
        cuda_lib.mkdir(parents=True, exist_ok=True)

        # Download and install cuSPARSELt locally (not system-wide)
        self._install_cusparselt_local(cuda_lib)

        # Modify activate script to include LD_LIBRARY_PATH
        self._patch_activate_script(cuda_lib)

        # Install PyTorch (must be first - includes its own numpy)
        # PyTorch has best GPU support on Jetson (sm_87, cuSPARSELt)
        self._install_pytorch_jetson()

        # Install remaining dependencies from requirements.txt
        self._install_from_requirements()

    def _install_pytorch_jetson(self):
        """Install PyTorch for Jetson based on JetPack version.

        PyTorch version is selected based on hardware/JetPack version.
        JetPack 6.x -> PyTorch 2.5.0 with sm_87 support
        """
        logger.info("Installing PyTorch for Jetson")

        # Detect JetPack version
        jetpack_version = self._get_jetpack_version()
        logger.info(f"Detected JetPack version: {jetpack_version}")

        # Select PyTorch wheel based on JetPack version
        # For JetPack 6.x (6.0, 6.1, 6.2) use PyTorch 2.5.0 with sm_87
        if jetpack_version and jetpack_version.startswith('6'):
            pytorch_wheel = (
                "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/"
                "torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
            )
        else:
            raise RuntimeError(
                f"Unsupported JetPack version: {jetpack_version}. "
                "This setup supports JetPack 6.x only."
            )

        logger.info(f"Installing PyTorch from: {pytorch_wheel}")

        # Install PyTorch (includes numpy, sympy, etc.)
        subprocess.run(
            [str(self.pip_path), 'install', pytorch_wheel],
            check=True
        )

        logger.info("PyTorch installed successfully")

    def _get_jetpack_version(self) -> str:
        """Get JetPack version from system.

        JetPack version is determined by the R# release number:
        - R35.x = JetPack 5.x
        - R36.x = JetPack 6.x
        """
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()
                # R36 = JetPack 6.x, R35 = JetPack 5.x
                if 'R36' in content:
                    return '6.0'  # JetPack 6.x
                elif 'R35' in content:
                    return '5.0'  # JetPack 5.x
        except (FileNotFoundError, PermissionError, IOError):
            pass
        return None

    def _install_other_deps(self):
        """Install other dependencies (fallback if no requirements.txt).

        PyTorch already includes: numpy, sympy, jinja2, networkx, etc.
        """
        logger.info("Installing other dependencies (fallback)")

        deps = [
            'pandas>=2.0.0',
            'pyarrow>=14.0.0',
            'scikit-learn>=1.3.0',
            'tqdm>=4.65.0',
        ]

        for dep in deps:
            logger.info(f"Installing {dep}")
            subprocess.run(
                [str(self.pip_path), 'install', dep],
                check=False  # Continue on individual failures
            )

        logger.info("All dependencies installed")

    def _install_cusparselt_local(self, cuda_lib: Path):
        """Install cuSPARSELt locally in venv (no sudo required).

        Jetson Orin with JetPack 6.x requires cuSPARSELt for PyTorch 2.4+.
        We install it locally in the venv, not system-wide.
        """
        logger.info("Installing cuSPARSELt locally (no sudo required)")

        cusparselt_version = "0.7.1.0"
        tarball = f"libcusparse_lt-linux-aarch64-{cusparselt_version}-archive.tar.xz"
        url = f"https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/{tarball}"

        import tempfile
        import urllib.request

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            tarpath = tmppath / tarball

            # Download
            logger.info(f"Downloading {tarball}")
            urllib.request.urlretrieve(url, tarpath)

            # Extract
            logger.info("Extracting cuSPARSELt")
            subprocess.run(
                ['tar', '-xf', str(tarpath), '-C', str(tmppath)],
                check=True
            )

            # Copy to venv
            extracted_dir = tmppath / f"libcusparse_lt-linux-aarch64-{cusparselt_version}-archive"
            if extracted_dir.exists():
                for lib_file in (extracted_dir / 'lib').glob('*.so*'):
                    dest = cuda_lib / lib_file.name
                    logger.info(f"Installing {lib_file.name} to {cuda_lib}")
                    import shutil
                    shutil.copy2(lib_file, dest)

    def _patch_activate_script(self, cuda_lib: Path):
        """Patch activate script to set LD_LIBRARY_PATH.

        This ensures cuSPARSELt is found when venv is activated.
        Note: For direct Python execution (without sourcing activate),
        users must set LD_LIBRARY_PATH manually:
            LD_LIBRARY_PATH=/path/to/venv/lib/cuda/lib python script.py
        """
        activate_script = self.venv_path / 'bin' / 'activate'

        if not activate_script.exists():
            logger.warning(f"Activate script not found: {activate_script}")
            return

        logger.info("Patching activate script for LD_LIBRARY_PATH")

        # Read current content
        content = activate_script.read_text()

        # Check if already patched
        if '_OLD_VIRTUAL_LD_LIBRARY_PATH' in content:
            logger.info("Activate script already patched")
            return

        # Patch deactivate function
        deactivate_patch = '''    if [ "${_OLD_VIRTUAL_LD_LIBRARY_PATH+set}" = "set" ] ; then
        LD_LIBRARY_PATH="${_OLD_VIRTUAL_LD_LIBRARY_PATH:-}"
        export LD_LIBRARY_PATH
        unset _OLD_VIRTUAL_LD_LIBRARY_PATH
    fi
'''

        content = content.replace(
            'unset _OLD_VIRTUAL_PYTHONHOME\n    fi',
            f'unset _OLD_VIRTUAL_PYTHONHOME\n    fi\n{deactivate_patch}'
        )

        # Patch activation section (after PATH setup)
        activate_patch = f'''
# Add cuSPARSELt library path for PyTorch (Jetson-specific)
_OLD_VIRTUAL_LD_LIBRARY_PATH="${{LD_LIBRARY_PATH:-}}"
if [ -d "{cuda_lib}" ] ; then
    LD_LIBRARY_PATH="{cuda_lib}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
    export LD_LIBRARY_PATH
fi
'''

        # Insert after PATH export
        content = content.replace(
            'export PATH\n\n# unset PYTHONHOME if set',
            f'export PATH\n{activate_patch}\n# unset PYTHONHOME if set'
        )

        # Write back
        activate_script.write_text(content)
        logger.info("Activate script patched successfully")

    def _install_from_requirements(self):
        """Install packages from model-core/requirements.txt.

        Skips packages already installed by PyTorch (numpy, etc.).
        Constrains numpy to <2.0 to maintain PyTorch compatibility.
        """
        req_file = Path(__file__).parent.parent.parent.parent / 'model-core' / 'requirements.txt'

        if not req_file.exists():
            logger.warning(f"Requirements file not found: {req_file}")
            # Fall back to hardcoded deps
            self._install_other_deps()
            return

        logger.info(f"Installing from {req_file}")

        # Packages already provided by PyTorch - skip these
        pytorch_packages = {'sympy', 'jinja2', 'networkx', 'fsspec',
                           'filelock', 'typing-extensions', 'mpmath', 'MarkupSafe'}

        # Note: numpy is special - PyTorch needs numpy<2, so we keep the one it installed
        # But pandas may try to upgrade it, so we need to constrain it

        # Read and filter requirements
        with open(req_file) as f:
            lines = f.readlines()

        filtered_reqs = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Extract package name (before version specifier)
            pkg_name = line.split('>=')[0].split('==')[0].split('<')[0].lower().replace('_', '-')
            if pkg_name == 'numpy':
                logger.info("Skipping numpy - using PyTorch's bundled version (<2.0)")
                continue
            if pkg_name not in pytorch_packages:
                filtered_reqs.append(line)
            else:
                logger.info(f"Skipping {pkg_name} - already provided by PyTorch")

        if not filtered_reqs:
            logger.info("No additional packages to install")
            return

        # Constrain numpy to prevent upgrades that break PyTorch
        logger.info("Pinning numpy<2 to maintain PyTorch compatibility")
        subprocess.run(
            [str(self.pip_path), 'install', '--upgrade', 'numpy<2'],
            check=False
        )

        # Install filtered packages
        for req in filtered_reqs:
            logger.info(f"Installing {req}")
            subprocess.run(
                [str(self.pip_path), 'install', req],
                check=False  # Continue on individual package failures
            )

        logger.info("Requirements installation complete")

    def _configure_cuda(self):
        """Configure venv for CUDA GPUs.

        Installs PyTorch with CUDA support (better than TensorFlow for most GPUs).
        """
        logger.info("Configuring for CUDA (PyTorch preferred over TensorFlow)")

        # Install PyTorch with CUDA support
        # PyTorch has better GPU support and easier installation than TensorFlow
        subprocess.run(
            [str(self.pip_path), 'install', 'torch', 'torchvision',
             '--index-url', 'https://download.pytorch.org/whl/cu121'],
            check=True
        )

        # Install remaining dependencies from requirements.txt
        self._install_from_requirements()

    def _configure_rocm(self):
        """Configure venv for ROCm GPUs (AMD).

        Installs PyTorch with ROCm support.
        """
        logger.info("Configuring for ROCm (AMD GPU)")

        # Install PyTorch with ROCm support
        subprocess.run(
            [str(self.pip_path), 'install', 'torch', 'torchvision',
             '--index-url', 'https://download.pytorch.org/whl/rocm5.7'],
            check=True
        )

        # Install remaining dependencies from requirements.txt
        self._install_from_requirements()

    def _configure_cpu(self):
        """Configure venv for CPU-only (no GPU).

        Installs PyTorch CPU version.
        """
        logger.info("Configuring for CPU-only")

        # Install PyTorch CPU version
        subprocess.run(
            [str(self.pip_path), 'install', 'torch', 'torchvision',
             '--index-url', 'https://download.pytorch.org/whl/cpu'],
            check=True
        )

        # Install remaining dependencies from requirements.txt
        self._install_from_requirements()


def create_venv_for_hardware(
    venv_path: str,
    output_config_path: Optional[str] = None
) -> tuple[Path, HardwareInfo]:
    """Create venv configured for detected hardware.

    Args:
        venv_path: Where to create venv
        output_config_path: Where to write hardware config JSON (optional)

    Returns:
        (venv_path, hardware_info)
    """
    # Detect hardware
    detector = HardwareDetector()
    hardware_info = detector.detect()

    # Create venv
    builder = VenvBuilder(venv_path, hardware_info)
    venv = builder.create()

    # Write hardware config
    if output_config_path:
        import json
        config_path = Path(output_config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(hardware_info.to_dict(), f, indent=2)
        logger.info(f"Hardware config written to {config_path}")

    return venv, hardware_info


if __name__ == '__main__':
    # CLI usage
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Create ML venv for detected hardware')
    parser.add_argument('venv_path', help='Path to create venv')
    parser.add_argument('--config', help='Path to write hardware config JSON')
    args = parser.parse_args()

    venv, hardware = create_venv_for_hardware(args.venv_path, args.config)

    print(f"\n✓ Virtual environment created at: {venv}")
    print(f"  Hardware: {hardware.gpu_type or 'CPU-only'}")
    print(f"  GPU: {hardware.gpu_name or 'N/A'}")
    print(f"\nTo activate:")
    print(f"  source {venv}/bin/activate")
