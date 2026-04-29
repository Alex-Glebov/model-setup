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
from .gpu_compatibility import test_gpu_compatibility, test_cpu_compatibility, test_tensorflow_compatibility
from .pip_version_checker import can_install_backend
from . import UNINSTALL_TIMEOUT, VERIFY_TIMEOUT


def is_wsl() -> bool:
    """Detect if running under Windows Subsystem for Linux.

    Returns:
        True if running in WSL environment
    """
    # Check for WSL in /proc/version
    try:
        with open('/proc/version', 'r') as f:
            version = f.read().lower()
            if 'microsoft' in version or 'wsl' in version:
                return True
    except (FileNotFoundError, PermissionError):
        pass

    # Check for WSL-specific environment variable
    if os.environ.get('WSL_DISTRO_NAME') or os.environ.get('WSL_INTEROP'):
        return True

    # Check for Windows-specific paths
    if Path('/mnt/c/Windows').exists():
        return True

    return False


def check_wsl_prerequisites() -> tuple[bool, list[str]]:
    """Check if WSL environment is properly configured for GPU.

    Returns:
        (is_ready, issues) where issues is a list of what's wrong
    """
    issues = []

    if not is_wsl():
        return True, []  # Not WSL, no issues

    logger.info("WSL environment detected")

    # Check WSL version (WSL 2 required for GPU)
    try:
        result = subprocess.run(['wsl.exe', '--version'],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            output = result.stdout.lower()
            if 'wsl version: 1' in output or 'wsl 1' in output:
                issues.append("WSL 1 detected - WSL 2 required for GPU support")
            else:
                logger.info("WSL 2 confirmed")
        else:
            # Try alternative check
            result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
            if 'microsoft' not in result.stdout.lower():
                issues.append("Cannot verify WSL version")
    except FileNotFoundError:
        issues.append("wsl.exe not found - are you in WSL?")
    except subprocess.TimeoutExpired:
        issues.append("WSL version check timeout")

    # Check for GPU support in WSL
    if not Path('/usr/lib/wsl/lib').exists():
        issues.append("WSL GPU support not detected (/usr/lib/wsl/lib missing)")

    # Check for DirectX
    if not Path('/usr/lib/wsl/lib/libdxcore.so').exists():
        issues.append("WSL DirectX support missing (libdxcore.so not found)")

    return len(issues) == 0, issues

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
        on_fail: str = 'n',  # 'a'=auto-delete, 'y'=ask, 'n'=keep
        install_all: bool = False  # Install all working backends
    ):
        self.venv_path = Path(venv_path)
        self.hardware_info = hardware_info
        self.on_fail = on_fail
        self.install_all = install_all
        self._pip_path: Optional[Path] = None
        self._log_file: Optional[Path] = None
        self._file_handler: Optional[logging.FileHandler] = None
        self._successful_backends: list[tuple[str, str]] = []  # (backend_name, install_type)

    @property
    def pip_path(self) -> Path:
        """Get path to venv's pip executable."""
        if self._pip_path is None:
            # Windows uses Scripts\pip.exe, Unix uses bin/pip
            if platform.system() == 'Windows':
                self._pip_path = self.venv_path / 'Scripts' / 'pip.exe'
            else:
                self._pip_path = self.venv_path / 'bin' / 'pip'
        return self._pip_path

    def create(self) -> tuple[Path, list[tuple[str, str]]]:
        """Create virtual environment with test-before-commit.

        In --all mode, installs multiple backends into ONE venv.
        Failed backends are uninstalled, successful ones remain.

        Returns:
            (venv_path, successful_backends) where successful_backends is
            list of (backend_name, install_type) tuples for all working backends
        """
        logger.info("=" * 60)
        logger.info("Venv Builder - Test Before Commit")

        # Report version and git branch
        try:
            import model_setup
            logger.info(f"Version: {model_setup.__version__}")
        except Exception:
            logger.info("Version: unknown")
        try:
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, timeout=5, cwd=str(Path(__file__).resolve().parent.parent.parent)
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
                logger.info(f"Branch: {branch}")
        except Exception:
            pass

        if self.install_all:
            logger.info("Mode: Install ALL working backends (--all)")
            logger.info("All backends will be installed into ONE venv")
        logger.info("=" * 60)

        # Preserve existing venv as venv.orig if it exists
        if self.venv_path.exists():
            orig_path = self._rename_to_orig()
            logger.info(f"Renamed existing venv to {orig_path}")

        # Create venv ONCE (all backends go into same venv)
        self._create_venv(self.venv_path)
        logger.info(f"Created venv at {self.venv_path}")

        # Build installation priority queue
        install_queue = self._build_install_queue()
        logger.info(f"Installation queue: {install_queue}")

        # Track successful backends
        self._successful_backends = []

        # Try each variant IN THE SAME VENV
        for backend_name, install_type in install_queue:
            logger.info(f"\nTrying {backend_name} ({install_type})...")

            # Install ML backend variant
            try:
                self._install_pytorch(install_type, backend_name)
            except Exception as e:
                logger.error(f"Install failed: {e}")
                logger.info(f"Skipping {backend_name} ({install_type})")
                continue

            # TEST: Verify GPU actually works
            logger.info(f"Testing {backend_name} ({install_type})...")
            success, msg = self._test_installation(install_type, backend_name)

            if success:
                logger.info(f"✓ {backend_name} ({install_type}) PASSED: {msg}")
                self._successful_backends.append((backend_name, install_type))

                if not self.install_all:
                    # Single install mode: install deps and return
                    self._install_remaining_deps()
                    self._cleanup_logging()
                    return self.venv_path, self._successful_backends
                # --all mode: continue installing more backends
            else:
                logger.warning(f"✗ {backend_name} ({install_type}) FAILED: {msg}")
                # Uninstall failed backend from shared venv
                self._uninstall_backend(backend_name, install_type)

        # Install remaining dependencies after all backends
        if self._successful_backends:
            self._install_remaining_deps()

        # Resolve nvidia package conflicts when multiple CUDA backends are installed
        if self.install_all and len(self._successful_backends) > 1:
            self._resolve_nvidia_conflicts()

        self._cleanup_logging()

        if self._successful_backends:
            return self.venv_path, self._successful_backends

        # Should never reach here (CPU always works)
        raise RuntimeError("All installation options failed including CPU")

    def _build_install_queue(self) -> list[tuple[str, str]]:
        """Build installation priority queue based on detected hardware.

        Returns list of (backend_name, install_type) tuples.
        Filters by PyPI availability and prerequisite checks.
        Logs detailed reasons why GPU options are skipped.
        """
        queue = []

        if not self.hardware_info:
            # Check if CPU backend is available on PyPI
            can_install, _ = can_install_backend('torch')
            if can_install:
                queue.append(('torch', 'cpu'))
            else:
                logger.error("No CPU backend available on PyPI - cannot proceed")
            return queue

        gpu_type = self.hardware_info.gpu_type

        # Check WSL prerequisites first
        if is_wsl():
            wsl_ready, wsl_issues = check_wsl_prerequisites()
            if not wsl_ready:
                logger.warning("WSL has issues that may prevent GPU usage:")
                for issue in wsl_issues:
                    logger.warning(f"  WSL Issue: {issue}")
                logger.info("Will attempt CPU fallback if GPU fails")

        # Map hardware to (backend_name, install_type) tuples
        # Priority: GPU backends first
        if gpu_type == 'jetson':
            # Check Jetson prerequisites
            can_proceed, missing = self._check_gpu_prerequisites('jetson')
            if not can_proceed:
                logger.warning("Jetson prerequisites missing:")
                for item in missing:
                    logger.warning(f"  - {item}")
                logger.info("Falling back to CPU")
            else:
                can_install, _ = can_install_backend('torch')
                if can_install:
                    queue.append(('torch', 'jetson'))
                else:
                    logger.warning("torch not available on PyPI for Jetson")

        elif gpu_type == 'cuda':
            # Check CUDA prerequisites
            can_proceed, missing = self._check_gpu_prerequisites('cuda')
            if not can_proceed:
                logger.warning("CUDA prerequisites missing:")
                for item in missing:
                    logger.warning(f"  - {item}")
                logger.info("Will attempt install anyway, but may fail")

            # Build backend list for CUDA
            cuda_backends = []
            can_install_torch, _ = can_install_backend('torch')
            if can_install_torch:
                cuda_backends.append(('torch', 'cuda'))
            else:
                logger.warning("torch not available on PyPI for CUDA")

            if self.install_all:
                can_install_tf, _ = can_install_backend('tensorflow')
                if can_install_tf:
                    cuda_backends.append(('tensorflow', 'cuda'))
                else:
                    logger.warning("tensorflow not available on PyPI for CUDA")

            # For --all: install TensorFlow FIRST so its nvidia packages take
            # precedence, then PyTorch with --force-reinstall resolves conflicts
            # without downgrading. Single install keeps torch first.
            if self.install_all and len(cuda_backends) > 1:
                cuda_backends.reverse()
                logger.info("--all mode: installing TensorFlow before PyTorch to minimize nvidia package conflicts")

            queue.extend(cuda_backends)

        elif gpu_type == 'rocm':
            # Check ROCm prerequisites
            can_proceed, missing = self._check_gpu_prerequisites('rocm')
            if not can_proceed:
                logger.warning("ROCm prerequisites missing:")
                for item in missing:
                    logger.warning(f"  - {item}")
                logger.info("Will attempt install anyway, but may fail")

            can_install, _ = can_install_backend('torch')
            if can_install:
                queue.append(('torch', 'rocm'))
            else:
                logger.warning("torch not available on PyPI for ROCm")

            # If --all, also try tensorflow
            if self.install_all:
                can_install_tf, _ = can_install_backend('tensorflow')
                if can_install_tf:
                    queue.append(('tensorflow', 'cpu'))  # TensorFlow doesn't support ROCm GPU
                    logger.info("Adding tensorflow (CPU) for --all mode")
                else:
                    logger.warning("tensorflow not available on PyPI")

        # CPU fallback using torch (lightest weight)
        can_install, _ = can_install_backend('torch')
        if can_install:
            if not queue:
                logger.info("No GPU backends available, using CPU")
            queue.append(('torch', 'cpu'))
        else:
            logger.error("No CPU backend available - installation cannot proceed")

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

    def _uninstall_backend(self, backend_name: str, install_type: str):
        """Uninstall a failed backend from the shared venv.

        Args:
            backend_name: 'torch' or 'tensorflow'
            install_type: 'cuda', 'rocm', 'cpu', 'jetson'
        """
        logger.info(f"Uninstalling {backend_name} ({install_type}) from shared venv...")

        try:
            if backend_name == 'torch':
                # Uninstall torch and related packages
                subprocess.run(
                    [str(self.pip_path), 'uninstall', '-y', 'torch', 'torchvision', 'torchaudio'],
                    capture_output=True, timeout=UNINSTALL_TIMEOUT
                )
            elif backend_name == 'tensorflow':
                # Uninstall tensorflow
                subprocess.run(
                    [str(self.pip_path), 'uninstall', '-y', 'tensorflow', 'tensorflow-cpu', 'tensorflow-gpu'],
                    capture_output=True, timeout=UNINSTALL_TIMEOUT
                )

            logger.info(f"Uninstalled {backend_name} ({install_type})")

        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout uninstalling {backend_name}")
        except Exception as e:
            logger.warning(f"Failed to uninstall {backend_name}: {e}")

    def _test_installation(self, install_type: str, backend_name: str = 'torch') -> tuple[bool, str]:
        """Test if installed ML backend is functional.

        Args:
            install_type: 'cuda', 'rocm', 'cpu', 'jetson'
            backend_name: 'torch' or 'tensorflow'

        Returns:
            (success, message) tuple
        """
        if backend_name == 'tensorflow':
            return test_tensorflow_compatibility(self.venv_path)
        elif install_type == 'cpu':
            return test_cpu_compatibility(self.venv_path)
        else:
            return test_gpu_compatibility(self.venv_path)

    def _install_pytorch(self, install_type: str, backend_name: str = 'torch'):
        """Install ML backend variant.

        Args:
            install_type: 'cuda', 'rocm', 'cpu', 'jetson'
            backend_name: 'torch' or 'tensorflow'
        """
        logger.info(f"Installing {backend_name} ({install_type})...")

        if install_type == 'jetson':
            self._install_pytorch_jetson()
        elif install_type == 'cuda':
            if backend_name == 'tensorflow':
                self._install_tensorflow_cuda()
            else:
                self._install_pytorch_cuda()
        elif install_type == 'rocm':
            self._install_pytorch_rocm()
        elif install_type == 'cpu':
            if backend_name == 'tensorflow':
                self._install_tensorflow_cpu()
            else:
                self._install_pytorch_cpu()
        else:
            raise ValueError(f"Unknown install type: {install_type}")

    def _install_pytorch_jetson(self):
        """Install PyTorch + Keras for Jetson.

        Uses HardwareInfo for JetPack version detection.
        """
        # Use detected version from hardware_info
        jetpack_version = self.hardware_info.cuda_version if self.hardware_info else None

        # Try to parse from /etc/nv_tegra_release as fallback
        if not jetpack_version:
            jetpack_version = self._read_jetpack_version_from_file()

        logger.info(f"Detected JetPack: {jetpack_version or 'unknown'}")

        # Determine wheel URL based on JetPack version
        pytorch_wheel = self._get_jetson_pytorch_wheel(jetpack_version)

        if not pytorch_wheel:
            raise RuntimeError(f"Unsupported JetPack version: {jetpack_version}")

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
        """Install PyTorch + Keras for CUDA.

        Uses detected CUDA version to select appropriate wheel.
        Falls back to cu121 if version detection fails.
        """
        cuda_version = self.hardware_info.cuda_version if self.hardware_info else None
        wheel_url = self._get_pytorch_wheel_url(cuda_version)

        logger.info(f"Installing PyTorch for CUDA {cuda_version or 'unknown'} (using {wheel_url})")

        subprocess.run(
            [str(self.pip_path), 'install', 'torch', 'torchvision',
             '--index-url', wheel_url],
            check=True
        )
        logger.info("PyTorch (CUDA) installed")

        # Install Keras 3.x
        subprocess.run(
            [str(self.pip_path), 'install', 'keras>=3.0.0'],
            check=True
        )
        logger.info("Keras installed")

    def _get_pytorch_wheel_url(self, cuda_version: Optional[str]) -> str:
        """Get PyTorch wheel URL based on detected CUDA version.

        Args:
            cuda_version: Detected CUDA version string (e.g., "12.2")

        Returns:
            PyTorch wheel index URL
        """
        if not cuda_version:
            logger.warning("CUDA version not detected, using default cu121")
            return "https://download.pytorch.org/whl/cu121"

        try:
            parts = cuda_version.split('.')
            major = parts[0]
            minor = parts[1] if len(parts) > 1 else '0'

            # Map to PyTorch wheel versions (last updated: April 2026)
            # PyTorch supports: cu118, cu121, cu124, cu126, cu128, etc.
            # Capped at cu124 for torch 2.5.1 compatibility.
            # Newer CUDA drivers (12.8+) are forward-compatible with cu124 wheels.
            cuda_num = int(major) * 10 + int(minor)

            if cuda_num >= 124:
                wheel_ver = "cu124"
            elif cuda_num >= 121:
                wheel_ver = "cu121"
            elif cuda_num >= 118:
                wheel_ver = "cu118"
            else:
                wheel_ver = "cu121"  # Minimum supported

            logger.info(f"Detected CUDA {cuda_version}, using PyTorch {wheel_ver}")
            return f"https://download.pytorch.org/whl/{wheel_ver}"

        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse CUDA version '{cuda_version}': {e}")
            return "https://download.pytorch.org/whl/cu121"

    def _install_pytorch_rocm(self):
        """Install PyTorch + Keras for ROCm.

        Uses ROCm version if detected, otherwise uses default.
        """
        rocm_version = self._get_rocm_version()
        wheel_url = self._get_rocm_wheel_url(rocm_version)

        logger.info(f"Installing PyTorch for ROCm {rocm_version or 'unknown'} (using {wheel_url})")

        subprocess.run(
            [str(self.pip_path), 'install', 'torch', 'torchvision',
             '--index-url', wheel_url],
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

    def _install_tensorflow_cuda(self):
        """Install TensorFlow + Keras for CUDA."""
        subprocess.run(
            [str(self.pip_path), 'install', 'tensorflow[and-cuda]'],
            check=True
        )
        logger.info("TensorFlow (CUDA) installed")

        # Keras is included with TensorFlow, but ensure >=3.0
        subprocess.run(
            [str(self.pip_path), 'install', 'keras>=3.0.0'],
            check=True
        )
        logger.info("Keras installed")

    def _install_tensorflow_cpu(self):
        """Install TensorFlow + Keras for CPU."""
        subprocess.run(
            [str(self.pip_path), 'install', 'tensorflow-cpu'],
            check=True
        )
        logger.info("TensorFlow (CPU) installed")

        # Keras is included with TensorFlow, but ensure >=3.0
        subprocess.run(
            [str(self.pip_path), 'install', 'keras>=3.0.0'],
            check=True
        )
        logger.info("Keras installed")

    def _resolve_nvidia_conflicts(self):
        """Resolve nvidia package conflicts after --all install.

        When both PyTorch and TensorFlow are installed, they may bring
        different versions of nvidia-* packages. This method reinstalls
        PyTorch with --no-deps to keep TensorFlow's newer nvidia packages
        (which are forward-compatible) while ensuring PyTorch's Python code
        is correctly installed.

        Logs warnings about expected pip dependency conflicts.
        """
        if not self.pip_path:
            return

        logger.info("Checking for nvidia package conflicts after --all install...")

        # Check if torch is installed and what CUDA variant it is
        try:
            result = subprocess.run(
                [str(self.pip_path), 'show', 'torch'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                logger.info("torch not installed, skipping conflict resolution")
                return

            torch_version = None
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    torch_version = line.split(':', 1)[1].strip()
                    break

            if not torch_version:
                return

            logger.info(f"torch version: {torch_version}")

            # Check for nvidia package conflicts via pip check
            check_result = subprocess.run(
                [str(self.pip_path), 'check'],
                capture_output=True, text=True, timeout=30
            )

            if check_result.returncode == 0 and not check_result.stdout:
                logger.info("No pip dependency conflicts found")
                return

            # Log conflicts but don't fail - CUDA is forward-compatible
            if check_result.stdout:
                for line in check_result.stdout.strip().split('\n'):
                    if 'nvidia-' in line:
                        logger.warning(f"Expected conflict: {line}")
                        logger.warning("  -> CUDA forward compatibility: this is usually harmless")
                    else:
                        logger.warning(f"pip check: {line}")

            # Determine correct wheel URL for reinstall
            cuda_version = self.hardware_info.cuda_version if self.hardware_info else None
            wheel_url = self._get_pytorch_wheel_url(cuda_version)

            logger.info(f"Reinstalling torch with --no-deps from {wheel_url} to resolve core installation...")
            subprocess.run(
                [str(self.pip_path), 'install', '--force-reinstall', '--no-deps',
                 'torch', 'torchvision', '--index-url', wheel_url],
                check=False, timeout=300
            )
            logger.info("torch reinstalled without changing nvidia packages")

        except subprocess.TimeoutExpired:
            logger.warning("Timeout checking/resolving nvidia conflicts")
        except Exception as e:
            logger.warning(f"Could not resolve nvidia conflicts: {e}")

    def _read_jetpack_version_from_file(self) -> Optional[str]:
        """Read JetPack version from /etc/nv_tegra_release.

        Uses R# (Jetson Linux version) to determine JetPack version:
        - R36 = JetPack 6.x
        - R35 = JetPack 5.x

        Returns:
            JetPack version string or None
        """
        try:
            with open('/etc/nv_tegra_release', 'r') as f:
                content = f.read()

                # Primary: Use R# (Jetson Linux version) for JetPack mapping
                # R36 = JetPack 6.x, R35 = JetPack 5.x
                if 'R36' in content:
                    # Check if it's 6.0 or 6.1 based on revision
                    if 'REVISION: 6.' in content:
                        return '6.1'
                    return '6.0'
                elif 'R35' in content:
                    # JetPack 5.x uses R35
                    if 'REVISION: 4.' in content:
                        return '5.1'
                    return '5.0'

        except (FileNotFoundError, PermissionError, IOError):
            pass
        return None

    def _get_jetson_pytorch_wheel(self, jetpack_version: Optional[str]) -> Optional[str]:
        """Get PyTorch wheel URL for Jetson based on JetPack version.

        Args:
            jetpack_version: JetPack version string (e.g., "6.0")

        Returns:
            PyTorch wheel URL or None if unsupported

        Note:
            JetPack 5.x wheels require Python 3.8
            JetPack 6.x wheels require Python 3.10
        """
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Check Python version compatibility
        if jetpack_version and jetpack_version.startswith("5.") and python_version != "3.8":
            logger.warning(f"JetPack {jetpack_version} wheels require Python 3.8, "
                          f"but running Python {python_version}")
            return None
        elif jetpack_version and jetpack_version.startswith("6.") and python_version != "3.10":
            logger.warning(f"JetPack {jetpack_version} wheels require Python 3.10, "
                          f"but running Python {python_version}")
            return None

        if not jetpack_version:
            logger.warning("JetPack version unknown, assuming 6.0")
            jetpack_version = "6.0"

        # Map JetPack versions to wheel URLs
        # These are NVIDIA-provided wheels for Jetson
        wheel_map = {
            # JetPack 6.x (Python 3.10)
            "6.0": (
                "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/"
                "torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
            ),
            "6.1": (
                "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/"
                "torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
            ),
            # JetPack 5.x (Python 3.8)
            "5.0": (
                "https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/"
                "torch-2.1.0a0+4136163.nv23.06-cp38-cp38-linux_aarch64.whl"
            ),
            "5.1": (
                "https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/"
                "torch-2.1.0a0+4136163.nv23.06-cp38-cp38-linux_aarch64.whl"
            ),
            # Add more versions as they become available
        }

        # Try exact match first
        if jetpack_version in wheel_map:
            return wheel_map[jetpack_version]

        # Try major version match
        major_ver = jetpack_version.split('.')[0]
        for ver, wheel in wheel_map.items():
            if ver.startswith(major_ver):
                logger.info(f"Using wheel for JetPack {ver} (closest match to {jetpack_version})")
                return wheel

        logger.error(f"No PyTorch wheel available for JetPack {jetpack_version}")
        return None

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

    def _get_rocm_version(self) -> Optional[str]:
        """Get ROCm version from system if available."""
        try:
            result = subprocess.run(['rocm-smi', '--showversion'],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Parse version from output
                for line in result.stdout.split('\n'):
                    if 'ROCm' in line and 'version' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if 'version' in part.lower() and i + 1 < len(parts):
                                return parts[i + 1].strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None

    def _get_rocm_wheel_url(self, rocm_version: Optional[str]) -> str:
        """Get PyTorch wheel URL for ROCm.

        Args:
            rocm_version: Detected ROCm version (e.g., "5.7")

        Returns:
            PyTorch wheel index URL
        """
        if not rocm_version:
            logger.warning("ROCm version not detected, using default rocm5.7")
            return "https://download.pytorch.org/whl/rocm5.7"

        try:
            parts = rocm_version.split('.')
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0

            # Map to PyTorch wheel versions
            if major >= 6:
                wheel_ver = "rocm6.0"
            elif major == 5 and minor >= 7:
                wheel_ver = "rocm5.7"
            elif major == 5 and minor >= 6:
                wheel_ver = "rocm5.6"
            else:
                wheel_ver = "rocm5.7"  # Default

            logger.info(f"Detected ROCm {rocm_version}, using PyTorch {wheel_ver}")
            return f"https://download.pytorch.org/whl/{wheel_ver}"

        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse ROCm version '{rocm_version}': {e}")
            return "https://download.pytorch.org/whl/rocm5.7"

    def _check_gpu_prerequisites(self, gpu_type: str) -> tuple[bool, list[str]]:
        """Check if GPU prerequisites are met and report what's missing.

        Args:
            gpu_type: 'cuda', 'rocm', 'jetson', or 'cpu'

        Returns:
            (can_proceed, missing_items) where missing_items is a list of
            human-readable descriptions of what's missing
        """
        missing = []

        if gpu_type == 'cuda':
            # Check for nvidia-smi (drivers)
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
                if result.returncode != 0:
                    missing.append("NVIDIA drivers (nvidia-smi failed)")
            except FileNotFoundError:
                missing.append("nvidia-smi not found - NVIDIA drivers not installed")
            except subprocess.TimeoutExpired:
                missing.append("nvidia-smi timeout - driver issue?")

            # Check for nvcc (CUDA toolkit)
            try:
                result = subprocess.run(['nvcc', '--version'], capture_output=True, timeout=5)
                if result.returncode != 0:
                    missing.append("CUDA toolkit (nvcc not working)")
            except FileNotFoundError:
                missing.append("CUDA toolkit not installed (nvcc not found)")

            # Check for cuDNN (optional but recommended)
            cudnn_path = Path('/usr/lib/x86_64-linux-gnu/libcudnn.so')
            if not cudnn_path.exists():
                # Also check other common locations
                alt_paths = [
                    Path('/usr/local/cuda/lib64/libcudnn.so'),
                    Path('/usr/lib/aarch64-linux-gnu/libcudnn.so'),
                ]
                if not any(p.exists() for p in alt_paths):
                    missing.append("cuDNN library (optional but recommended)")

        elif gpu_type == 'rocm':
            # Check for rocm-smi
            try:
                result = subprocess.run(['rocm-smi'], capture_output=True, timeout=5)
                if result.returncode != 0:
                    missing.append("ROCm drivers (rocm-smi failed)")
            except FileNotFoundError:
                missing.append("ROCm not installed (rocm-smi not found)")

            # Check for hipcc
            try:
                result = subprocess.run(['hipcc', '--version'], capture_output=True, timeout=5)
                if result.returncode != 0:
                    missing.append("HIP compiler (hipcc not working)")
            except FileNotFoundError:
                missing.append("HIP toolkit not installed (hipcc not found)")

        elif gpu_type == 'jetson':
            # Check for JetPack
            if not Path('/etc/nv_tegra_release').exists():
                missing.append("JetPack not detected (/etc/nv_tegra_release missing)")

            # Check for JetPack version compatibility
            jetpack = self.hardware_info.cuda_version if self.hardware_info else None
            if jetpack and not jetpack.startswith('6'):
                missing.append(f"JetPack version {jetpack} (6.x required)")

        # Log findings
        if missing:
            logger.warning(f"GPU prerequisites check for {gpu_type}:")
            for item in missing:
                logger.warning(f"  - Missing: {item}")
        else:
            logger.info(f"All {gpu_type} prerequisites satisfied")

        return len(missing) == 0, missing


def create_venv_for_hardware(
    venv_path: str,
    output_config_path: Optional[str] = None,
    on_fail: str = 'n',
    install_all: bool = False
) -> tuple[Path, HardwareInfo, str, list[tuple[str, str]]]:
    """Create venv configured for detected hardware.

    Args:
        venv_path: Where to create venv
        output_config_path: Where to write hardware config JSON (optional)
        on_fail: Action on failed install - 'a'=auto-delete, 'y'=ask, 'n'=keep
        install_all: If True, install all working backends (not just priority)

    Returns:
        (venv_path, hardware_info, keras_backend, all_successful_backends)
    """
    # Detect hardware
    detector = HardwareDetector()
    hardware_info = detector.detect()

    # Create venv with test-before-commit
    builder = VenvBuilder(venv_path, hardware_info, on_fail=on_fail, install_all=install_all)
    venv, successful_backends = builder.create()

    # Determine primary keras_backend from successful installs or hardware_info
    keras_backend_map = {
        'pytorch': 'torch',
        'tensorflow': 'tensorflow',
        'cpu': 'torch',
    }

    if successful_backends:
        # Use first successful backend as primary
        keras_backend = successful_backends[0][0]
    else:
        # Fallback to hardware detection
        keras_backend = keras_backend_map.get(hardware_info.preferred_backend, 'torch')

    # Write hardware config with keras_backend
    if output_config_path:
        import json
        config_path = Path(output_config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = hardware_info.to_dict()
        config['keras_backend'] = keras_backend
        config['available_backends'] = [b[0] for b in successful_backends]

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Hardware config written to {config_path}")
        logger.info(f"Primary Keras backend: {keras_backend}")
        if len(successful_backends) > 1:
            logger.info(f"Alternative backends available: {[b[0] for b in successful_backends[1:]]}")

    # Generate keras_backend.py for model-core with all available backends
    _generate_keras_backend_py(venv_path, keras_backend, successful_backends)

    # Run verification
    _verify_installation(venv)

    return venv, hardware_info, keras_backend, successful_backends


def _generate_keras_backend_py(
    venv_path,
    primary_backend: str,
    all_backends: list[tuple[str, str]] = None
):
    """Generate keras_backend.py for model-core package.

    This file sets KERAS_BACKEND environment variable before importing keras.
    Other modules import keras from here instead of directly.

    When multiple backends are available, generates commented lines for easy switching.

    Args:
        venv_path: Path to venv (contains model-core checkout)
        primary_backend: Primary Keras backend name to use by default
        all_backends: List of (backend_name, install_type) tuples for all working backends
    """
    # Find model_core directory in the venv's parent (assumes model-core is checked out there)
    model_core_parent = Path(venv_path).parent
    keras_backend_path = model_core_parent / 'model_core' / 'keras_backend.py'

    # Build the backend configuration lines
    backend_lines = []

    # Primary backend (active)
    backend_lines.append(f'# Active backend: {primary_backend}')
    backend_lines.append(f'os.environ["KERAS_BACKEND"] = "{primary_backend}"')

    # Add commented alternatives if --all mode was used
    if all_backends and len(all_backends) > 1:
        backend_lines.append('')
        backend_lines.append('# Alternative backends (uncomment to switch):')

        for backend_name, install_type in all_backends:
            if backend_name != primary_backend:
                backend_lines.append(f'# Backend: {backend_name} ({install_type})')
                backend_lines.append(f'# os.environ["KERAS_BACKEND"] = "{backend_name}"')

    backend_config = '\n'.join(backend_lines)

    content = f'''\"\"\"Keras backend initialization.

Auto-generated by model-setup. Do not edit manually.
Primary backend: {primary_backend}
\"\"\"

import os

# Set Keras backend BEFORE importing keras
{backend_config}

# Import keras with configured backend
import keras

__all__ = ["keras"]
'''

    try:
        keras_backend_path.parent.mkdir(parents=True, exist_ok=True)
        with open(keras_backend_path, 'w') as f:
            f.write(content)
        logger.info(f"Generated {keras_backend_path}")
        if all_backends and len(all_backends) > 1:
            logger.info(f"Available backends: {[b[0] for b in all_backends]}")
            logger.info(f"Switch backends by editing: {keras_backend_path}")
    except Exception as e:
        logger.warning(f"Could not generate keras_backend.py: {e}")


def _verify_installation(venv_path: Path) -> bool:
    """Verify installed backends are working.

    Runs verify.py using venv Python without activating.
    Logs results and returns overall success status.

    Args:
        venv_path: Path to virtual environment

    Returns:
        True if all backends working, False otherwise
    """
    verify_script = Path(__file__).parent / 'verify.py'
    if not verify_script.exists():
        logger.warning(f"Verification script not found: {verify_script}")
        return True  # Don't fail installation if verify script missing

    python_path = venv_path / 'bin' / 'python'
    if not python_path.exists():
        python_path = venv_path / 'Scripts' / 'python.exe'  # Windows

    if not python_path.exists():
        logger.warning(f"Python not found in venv: {venv_path}")
        return True

    logger.info("Running verification checks...")

    try:
        result = subprocess.run(
            [str(python_path), str(verify_script)],
            capture_output=True,
            text=True,
            timeout=VERIFY_TIMEOUT
        )

        # Log output
        for line in result.stdout.split('\n'):
            if line.strip():
                logger.info(f"  {line}")

        if result.stderr:
            for line in result.stderr.split('\n'):
                if line.strip():
                    logger.warning(f"  {line}")

        success = result.returncode == 0
        if success:
            logger.info("✓ Verification PASSED")
        else:
            logger.warning("✗ Verification FAILED")

        return success

    except subprocess.TimeoutExpired:
        logger.warning(f"Verification timed out after {VERIFY_TIMEOUT} seconds")
        return False
    except Exception as e:
        logger.warning(f"Verification failed: {e}")
        return False


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
    parser.add_argument('--all', action='store_true', dest='install_all',
                        help='Install ALL working backends with commented switch options')
    args = parser.parse_args()

    venv, hardware, keras_backend, all_backends = create_venv_for_hardware(
        args.venv_path, args.config, args.on_fail, args.install_all
    )

    print(f"\n✓ Virtual environment created at: {venv}")
    print(f"  Hardware: {hardware.gpu_type or 'CPU-only'}")
    print(f"  GPU: {hardware.gpu_name or 'N/A'}")
    print(f"  Primary Keras Backend: {keras_backend}")

    if len(all_backends) > 1:
        print(f"\n  Available backends:")
        for i, (backend_name, install_type) in enumerate(all_backends):
            marker = " (active)" if i == 0 else ""
            print(f"    - {backend_name} ({install_type}){marker}")
        print(f"\n  Switch backends by editing: model_core/keras_backend.py")

    print(f"\nTo activate:")
    if platform.system() == 'Windows':
        print(f"  {venv}\\Scripts\\activate")
    else:
        print(f"  source {venv}/bin/activate")
