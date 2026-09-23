"""Virtual environment builder for ML training.

Creates the target venv with hardware-specific configuration using
probe-then-commit: framework candidates are probed inside a disposable
probe venv and only the proven package set is committed into the target
venv (which is never deleted).
"""

import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
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
    except subprocess.TimeoutExpired:
        issues.append("WSL version check timeout")
    except (OSError, subprocess.SubprocessError):
        # OSError covers FileNotFoundError AND PermissionError (e.g. Windows
        # binaries resolved through WSL PATH interop)
        issues.append("wsl.exe not found - are you in WSL?")

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

    Implements probe-then-commit strategy:
    1. Probe candidates inside a disposable probe venv (never the target)
    2. Only the proven package set is committed into the target venv
    3. The target venv is never deleted - created if missing, updated
       in place otherwise (pip skips already-satisfied requirements, a
       matching working backend is not reinstalled)
    4. CPU is guaranteed fallback
    """

    def __init__(
        self,
        venv_path: str,
        hardware_info: Optional[HardwareInfo] = None,
        on_fail: str = 'n',  # legacy option, kept for CLI compatibility
        install_all: bool = False,  # Install all working backends
        probe_venv_path: Optional[str] = None,  # Disposable probe venv
        requirements_path: Optional[str] = None  # Explicit requirements.txt
    ):
        self.venv_path = Path(venv_path)
        self.hardware_info = hardware_info
        self.on_fail = on_fail  # legacy option, kept for CLI compatibility
        self.install_all = install_all
        # Explicit path to the project's requirements.txt; None = auto-discover
        # (see _install_remaining_deps)
        self.requirements_path = requirements_path
        # Optional disposable probe venv: candidates and project
        # requirements are probed inside it; only the proven set is
        # committed into self.venv_path. None = legacy single-venv mode
        # (candidates probed directly in the target venv).
        self.probe_venv_path = Path(probe_venv_path) if probe_venv_path else None
        # Venv that pip/uninstall/compat-test operations currently target
        self._active_venv = self.venv_path
        self._pip_path: Optional[Path] = None
        self._log_file: Optional[Path] = None
        self._file_handler: Optional[logging.FileHandler] = None
        self._successful_backends: list[tuple[str, str]] = []  # (backend_name, install_type)
        # Exact package versions validated during probing, per backend name.
        # The commit phase re-installs these pins so the target venv ends up
        # with the same versions that were actually verified (prevents
        # probe/commit version drift).
        self._probe_pins: dict[str, dict[str, str]] = {}
        # Pins currently applied by the install methods ({} while probing,
        # the probed pins while committing)
        self._active_pins: dict[str, str] = {}

    @property
    def pip_path(self) -> Path:
        """Get path to the active venv's pip executable."""
        if self._pip_path is None:
            # Windows uses Scripts\pip.exe, Unix uses bin/pip
            if platform.system() == 'Windows':
                self._pip_path = self._active_venv / 'Scripts' / 'pip.exe'
            else:
                self._pip_path = self._active_venv / 'bin' / 'pip'
        return self._pip_path

    def _use_venv(self, venv_path: Path):
        """Point pip/uninstall/compat-test operations at a specific venv."""
        self._active_venv = Path(venv_path)
        self._pip_path = None  # recompute for the newly active venv

    def _validate_existing_venv(self, venv_path: Path):
        """Fail fast if an existing venv cannot be reused on this platform.

        A venv is platform-specific: Windows uses Scripts\\pip.exe while
        Linux/macOS/WSL use bin/pip, and framework wheels are compiled
        per platform. Reusing a venv created on the other platform
        (typically Windows Python vs WSL sharing one folder) would
        otherwise fail much later with a cryptic FileNotFoundError.
        The venv is never deleted automatically - this only reports it.
        """
        if platform.system() == 'Windows':
            current_pip = venv_path / 'Scripts' / 'pip.exe'
            foreign_pip = venv_path / 'bin' / 'pip'
            current_side = 'Windows'
            foreign_side = 'Linux/macOS/WSL'
        else:
            current_pip = venv_path / 'bin' / 'pip'
            foreign_pip = venv_path / 'Scripts' / 'pip.exe'
            current_side = 'Linux/macOS/WSL'
            foreign_side = 'Windows'

        if current_pip.exists():
            return  # platform layout matches - safe to reuse

        if foreign_pip.exists():
            message = (
                f"Existing venv '{venv_path}' was created by {foreign_side} "
                f"Python (pip found at '{foreign_pip}') and cannot be "
                f"reused from {current_side} (this platform expects "
                f"'{current_pip}'). Virtual environments are "
                "platform-specific. Re-run with a separate venv path for "
                f"this platform, or delete '{venv_path}' and re-run to "
                "rebuild it for this platform."
            )
            logger.error(message)
            raise RuntimeError(message)

        message = (
            f"Existing venv '{venv_path}' has no usable pip at "
            f"'{current_pip}' (corrupted, created without pip, or not a "
            f"real venv). Delete '{venv_path}' and re-run to rebuild it."
        )
        logger.error(message)
        raise RuntimeError(message)

    def create(self) -> tuple[Path, list[tuple[str, str]]]:
        """Create or update the target venv using probe-then-commit.

        Probe phase (with a probe venv): candidates and project requirements
        are installed and tested inside the disposable probe venv; failed
        candidates never touch the target. Legacy mode (no probe venv):
        candidates are probed directly inside the target venv (created if
        missing). In BOTH modes the target venv is never deleted.

        Returns:
            (venv_path, successful_backends) where successful_backends is a
            list of (backend_name, install_type) tuples for all working backends
        """
        # Harden every child process (pip, verify.py) against console encoding
        # crashes (Windows cp1252 cannot encode all Unicode) and silence pip's
        # self-upgrade notices. setdefault keeps user overrides intact.
        os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
        os.environ.setdefault('PYTHONUTF8', '1')
        os.environ.setdefault('PIP_DISABLE_PIP_VERSION_CHECK', '1')

        logger.info("=" * 60)
        logger.info("Venv Builder - Probe Then Commit")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")

        if self.install_all:
            logger.info("Mode: Install ALL working backends (--all)")
            logger.info("All backends will be installed into ONE venv")

        if self.probe_venv_path is not None:
            logger.info(f"Probe venv (disposable): {self.probe_venv_path}")
        logger.info(f"Target venv (persistent - never deleted): {self.venv_path}")
        logger.info("=" * 60)

        # Fail fast, before any probing: if the target venv already
        # exists it must be usable from this platform (a Windows venv
        # cannot be updated from WSL and vice versa). Without this check
        # the whole probe phase would run first and the mismatch would
        # only surface later, when the commit phase first invokes pip.
        if self.venv_path.exists():
            self._validate_existing_venv(self.venv_path)

        if self.probe_venv_path is not None:
            # ---- Phase 1: probe candidates in the disposable probe venv ----
            self._use_venv(self.probe_venv_path)
            if not self.probe_venv_path.exists():
                self._create_venv(self.probe_venv_path)
                logger.info(f"Created probe venv at {self.probe_venv_path}")
            else:
                logger.info(f"Reusing probe venv at {self.probe_venv_path}")
                self._validate_existing_venv(self.probe_venv_path)
            # Full-fidelity dry run: same requirements pass as the commit phase
            self._install_remaining_deps()
            self._successful_backends = self._probe_candidates()
        else:
            # Legacy single-venv mode: probe directly in the target venv
            if self.venv_path.exists():
                logger.info(f"Reusing existing venv at {self.venv_path} (never deleted; updated in place)")
                self._validate_existing_venv(self.venv_path)
            else:
                self._create_venv(self.venv_path)
                logger.info(f"Created venv at {self.venv_path}")
            self._use_venv(self.venv_path)
            self._successful_backends = self._probe_candidates()

        # ---- Phase 2: commit the proven set into the target venv ----
        self._commit_backends(self._successful_backends)

        self._cleanup_logging()

        if self._successful_backends:
            return self.venv_path, self._successful_backends

        # Should never reach here (CPU always works)
        raise RuntimeError("All installation options failed including CPU")

    def _probe_candidates(self) -> list[tuple[str, str]]:
        """Try each install candidate inside the active (probe) venv.

        Failed candidates are uninstalled so the next candidate starts
        clean. In single-install mode probing stops at the first working
        backend; in --all mode every candidate is probed.
        """
        install_queue = self._build_install_queue()
        logger.info(f"Installation queue: {install_queue}")

        winners: list[tuple[str, str]] = []
        for backend_name, install_type in install_queue:
            logger.info(f"\nTrying {backend_name} ({install_type})...")

            # Install ML backend variant
            try:
                self._install_pytorch(install_type, backend_name)
            except Exception as e:
                logger.error(f"Install failed: {e}")
                logger.info(f"Skipping {backend_name} ({install_type})")
                continue

            # TEST: Verify backend actually works
            logger.info(f"Testing {backend_name} ({install_type})...")
            success, msg = self._test_installation(install_type, backend_name)

            if success:
                logger.info(f"[OK] {backend_name} ({install_type}) PASSED: {msg}")
                winners.append((backend_name, install_type))
                # Record exact versions so the commit phase can reproduce them
                self._record_probe_pins(backend_name)
                if not self.install_all:
                    break
            else:
                logger.warning(f"[FAIL] {backend_name} ({install_type}) FAILED: {msg}")
                # Uninstall the failed candidate so the probe env stays clean
                self._uninstall_backend(backend_name, install_type)

        return winners

    def _record_probe_pins(self, backend_name: str):
        """Record the exact package versions that just passed probing.

        The commit phase re-installs these pins so a pre-existing target
        venv is upgraded to the versions that were actually validated
        (prevents probe/commit version drift: without this, pip sees an
        'already satisfied' requirement in the target and keeps the old
        version while the probe validated a newer one).
        """
        packages = ('torch', 'torchvision', 'tensorflow', 'tensorflow-cpu', 'keras')
        try:
            result = subprocess.run(
                [str(self.pip_path), 'freeze'],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return
            pins = {}
            for line in result.stdout.split('\n'):
                if '==' in line and '@' not in line:
                    name, _, version = line.partition('==')
                    if name.strip().lower() in packages:
                        pins[name.strip().lower()] = version.strip()
            if pins:
                self._probe_pins[backend_name] = pins
                logger.info(f"Probed versions to commit for {backend_name}: {pins}")
        except (subprocess.SubprocessError, OSError) as e:
            logger.debug(f"Could not record probe pins for {backend_name}: {e}")

    def _pin(self, package: str, default_spec: str) -> str:
        """Return 'package==probed_version' if probed, else the default spec."""
        version = self._active_pins.get(package)
        if version:
            return f"{package}=={version}"
        return default_spec

    def _version_matches_pins(self, backend_name: str, installed_version: str) -> bool:
        """Does an installed backend version match what the probe validated?

        When nothing was pinned (legacy single-venv mode, or pin recording
        failed) any working installation is treated as matching.
        """
        pin = self._probe_pins.get(backend_name, {}).get(backend_name)
        if pin is None:
            return True
        return installed_version == pin

    def _commit_backends(self, winners: list[tuple[str, str]]):
        """Commit the proven package set into the target venv (idempotent).

        The target is created if missing and updated in place otherwise.
        Project requirements are re-installed via pip (already-satisfied
        packages are skipped), then each winning backend is committed:
        a matching working flavor is kept, a conflicting flavor is replaced.
        """
        self._use_venv(self.venv_path)
        if self.venv_path.exists():
            logger.info(f"Reusing existing venv at {self.venv_path} (never deleted; updated in place)")
            self._validate_existing_venv(self.venv_path)
            self._setup_venv_logging(self.venv_path)
        else:
            self._create_venv(self.venv_path)
            logger.info(f"Created venv at {self.venv_path}")

        # Layer 1: project requirements (pip skips already-satisfied packages)
        self._install_remaining_deps()

        # Layer 2: proven ML framework winners
        for backend_name, install_type in winners:
            # Commit the exact versions that were validated in the probe
            self._active_pins = self._probe_pins.get(backend_name, {})
            installed_version = self._detect_backend_version(backend_name)
            if installed_version is not None:
                if self._backend_matches(installed_version, backend_name, install_type):
                    if self._version_matches_pins(backend_name, installed_version):
                        ok, msg = self._test_installation(install_type, backend_name)
                        if ok:
                            logger.info(f"[OK] {backend_name} ({install_type}) already installed "
                                        f"at the probed version and working in target - "
                                        f"skipping reinstall ({msg})")
                            continue
                        logger.warning(f"{backend_name} present in target but not working "
                                       f"({msg}) - replacing")
                    else:
                        logger.info(f"Upgrading {backend_name} in target from "
                                    f"'{installed_version}' to the probed version "
                                    f"({self._active_pins.get(backend_name, 'latest')})")
                else:
                    logger.info(f"Replacing {backend_name} flavor in target "
                                f"('{installed_version}') with {install_type}")
                self._uninstall_backend(backend_name, install_type)

            self._install_pytorch(install_type, backend_name)
            ok, msg = self._test_installation(install_type, backend_name)
            if ok:
                logger.info(f"[OK] Committed {backend_name} ({install_type}) into target: {msg}")
            else:
                logger.error(f"Committed {backend_name} ({install_type}) failed "
                             f"verification in target: {msg}")

        # Resolve nvidia package conflicts when multiple CUDA backends installed
        if self.install_all and len(winners) > 1:
            self._resolve_nvidia_conflicts()

    def _detect_backend_version(self, backend_name: str) -> Optional[str]:
        """Installed version of a backend in the target venv (None = absent)."""
        if backend_name == 'tensorflow':
            probe = "import tensorflow as tf; print(tf.__version__)"
        else:
            probe = "import torch; print(torch.__version__)"
        if platform.system() == 'Windows':
            python = self.venv_path / 'Scripts' / 'python.exe'
        else:
            python = self.venv_path / 'bin' / 'python'
        try:
            result = subprocess.run(
                [str(python), '-c', probe],
                capture_output=True, text=True, timeout=VERIFY_TIMEOUT
            )
            if result.returncode != 0:
                return None
            output = result.stdout.strip()
            return output.splitlines()[-1] if output else None
        except Exception as e:
            logger.warning(f"Could not probe {backend_name} in target venv: {e}")
            return None

    def _backend_matches(self, installed_version: str, backend_name: str,
                          install_type: str) -> bool:
        """Does an installed backend match the requested install type?

        TensorFlow wheels carry no local flavor tag, so any working
        installation matches (the compatibility test decides). For torch
        the local version tag identifies the flavor.
        """
        if backend_name == 'tensorflow':
            return True
        flavor_tags = {'cpu': '+cpu', 'rocm': '+rocm', 'cuda': '+cu', 'jetson': 'nv'}
        tag = flavor_tags.get(install_type)
        if tag is None:
            return True
        return tag in installed_version

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
        # UTF-8 log file: never crashes on non-ASCII (e.g. Unicode GPU names)
        # regardless of the console's code page
        self._file_handler = logging.FileHandler(self._log_file, encoding='utf-8')
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
            return test_tensorflow_compatibility(self._active_venv)
        elif install_type == 'cpu':
            return test_cpu_compatibility(self._active_venv)
        else:
            return test_gpu_compatibility(self._active_venv)

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

        # Install Keras 3.x (pinned to the probed version when available)
        subprocess.run(
            [str(self.pip_path), 'install', self._pin('keras', 'keras>=3.0.0')],
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
            [str(self.pip_path), 'install',
             self._pin('torch', 'torch'),
             self._pin('torchvision', 'torchvision'),
             '--index-url', wheel_url],
            check=True
        )
        logger.info("PyTorch (CUDA) installed")

        # Install Keras 3.x (pinned to the probed version when available)
        subprocess.run(
            [str(self.pip_path), 'install', self._pin('keras', 'keras>=3.0.0')],
            check=True
        )
        logger.info("Keras installed")

    # CUDA driver version (as major*10+minor) -> PyTorch wheel index.
    # First matching row wins, so newer drivers get the newest index.
    # Verified against download.pytorch.org (September 2026): cu130 serves
    # torch 2.14.0, cu126/cu128 serve 2.9.1, cu124 serves 2.6.0.
    CUDA_WHEEL_MAP = [
        (130, "cu130"),
        (128, "cu128"),
        (126, "cu126"),
        (124, "cu124"),
        (121, "cu121"),
        (118, "cu118"),
    ]
    # Conservative fallback when detection fails (also safe on older GPUs:
    # cu121 wheels support sm_37+)
    DEFAULT_CUDA_WHEEL = "cu121"
    # Environment variable for an explicit index override, e.g.
    # MODEL_SETUP_TORCH_INDEX=https://download.pytorch.org/whl/cu126
    TORCH_INDEX_OVERRIDE_ENV = "MODEL_SETUP_TORCH_INDEX"

    def _get_pytorch_wheel_url(self, cuda_version: Optional[str]) -> str:
        """Get PyTorch wheel URL based on the detected CUDA driver version.

        Preference order:
          1. MODEL_SETUP_TORCH_INDEX environment variable (explicit override)
          2. Highest supported wheel index for the detected driver version
          3. DEFAULT_CUDA_WHEEL when version detection fails

        Args:
            cuda_version: Detected CUDA version string (e.g., "13.2")

        Returns:
            PyTorch wheel index URL
        """
        override = os.environ.get(self.TORCH_INDEX_OVERRIDE_ENV)
        if override:
            logger.info(f"Using PyTorch index from {self.TORCH_INDEX_OVERRIDE_ENV}: {override}")
            return override

        if not cuda_version:
            logger.warning(f"CUDA version not detected, using default {self.DEFAULT_CUDA_WHEEL}")
            return f"https://download.pytorch.org/whl/{self.DEFAULT_CUDA_WHEEL}"

        try:
            parts = cuda_version.split('.')
            cuda_num = int(parts[0]) * 10 + int(parts[1] if len(parts) > 1 else '0')

            # Highest supported index for this driver (or the oldest index
            # for drivers below every threshold - cu118 covers CUDA 11.x)
            wheel_ver = next(
                (wheel for threshold, wheel in self.CUDA_WHEEL_MAP
                 if cuda_num >= threshold),
                self.CUDA_WHEEL_MAP[-1][1],
            )

            logger.info(f"Detected CUDA {cuda_version}, using PyTorch {wheel_ver}")
            return f"https://download.pytorch.org/whl/{wheel_ver}"

        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse CUDA version '{cuda_version}': {e}")
            return f"https://download.pytorch.org/whl/{self.DEFAULT_CUDA_WHEEL}"

    def _install_pytorch_rocm(self):
        """Install PyTorch + Keras for ROCm.

        Uses ROCm version if detected, otherwise uses default.
        """
        rocm_version = self._get_rocm_version()
        wheel_url = self._get_rocm_wheel_url(rocm_version)

        logger.info(f"Installing PyTorch for ROCm {rocm_version or 'unknown'} (using {wheel_url})")

        subprocess.run(
            [str(self.pip_path), 'install',
             self._pin('torch', 'torch'),
             self._pin('torchvision', 'torchvision'),
             '--index-url', wheel_url],
            check=True
        )
        logger.info("PyTorch (ROCm) installed")

        # Install Keras 3.x (pinned to the probed version when available)
        subprocess.run(
            [str(self.pip_path), 'install', self._pin('keras', 'keras>=3.0.0')],
            check=True
        )
        logger.info("Keras installed")

    def _install_pytorch_cpu(self):
        """Install PyTorch + Keras for CPU."""
        subprocess.run(
            [str(self.pip_path), 'install',
             self._pin('torch', 'torch'),
             self._pin('torchvision', 'torchvision'),
             '--index-url', 'https://download.pytorch.org/whl/cpu'],
            check=True
        )
        logger.info("PyTorch (CPU) installed")

        # Install Keras 3.x (pinned to the probed version when available)
        subprocess.run(
            [str(self.pip_path), 'install', self._pin('keras', 'keras>=3.0.0')],
            check=True
        )
        logger.info("Keras installed")

    def _install_tensorflow_cuda(self):
        """Install TensorFlow + Keras for CUDA."""
        subprocess.run(
            [str(self.pip_path), 'install',
             self._pin('tensorflow', 'tensorflow[and-cuda]')],
            check=True
        )
        logger.info("TensorFlow (CUDA) installed")

        # Keras is included with TensorFlow, but ensure >=3.0
        subprocess.run(
            [str(self.pip_path), 'install', self._pin('keras', 'keras>=3.0.0')],
            check=True
        )
        logger.info("Keras installed")

    def _install_tensorflow_cpu(self):
        """Install TensorFlow + Keras for CPU."""
        subprocess.run(
            [str(self.pip_path), 'install',
             self._pin('tensorflow-cpu', 'tensorflow-cpu')],
            check=True
        )
        logger.info("TensorFlow (CPU) installed")

        # Keras is included with TensorFlow, but ensure >=3.0
        subprocess.run(
            [str(self.pip_path), 'install', self._pin('keras', 'keras>=3.0.0')],
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

        # Locate requirements.txt by convention - no hardcoded repository layout:
        #   1. explicitly provided requirements path (requirements_path)
        #   2. next to the target venv (the venv is created inside the project root)
        #   3. in the current working directory (the script is run from the project root)
        #   4. next to the invoking script (e.g. test_venv_builder.py in the
        #      project root, invoked from anywhere)
        candidates = []
        if self.requirements_path is not None:
            candidates.append(Path(self.requirements_path).resolve())
        candidates.extend([
            Path(self.venv_path).resolve().parent / 'requirements.txt',
            Path.cwd() / 'requirements.txt',
        ])
        try:
            script_dir = Path(sys.argv[0]).resolve().parent
            if script_dir not in (Path.cwd(), Path(self.venv_path).resolve().parent):
                candidates.append(script_dir / 'requirements.txt')
        except (OSError, IndexError):
            pass
        # De-duplicate while preserving order
        candidates = list(dict.fromkeys(candidates))

        req_file = next((c for c in candidates if c.is_file()), None)

        if req_file is None:
            # A missing requirements file silently produced incomplete venvs;
            # this is a real error, not a cosmetic warning
            logger.error(
                "requirements.txt not found - project dependencies will NOT be "
                f"installed. Looked in: {[str(c) for c in candidates]}"
            )
            return

        if self.requirements_path is not None:
            logger.info(f"Using requirements file: {req_file}")

        # Skip packages bundled with PyTorch
        pytorch_packages = {'sympy', 'jinja2', 'networkx', 'fsspec',
                           'filelock', 'typing-extensions', 'mpmath', 'MarkupSafe'}

        with open(req_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            pkg_name = (line.split('>=')[0].split('==')[0].split('<')[0]
                        .split('~=')[0].split('[')[0].lower().replace('_', '-'))

            if pkg_name not in pytorch_packages:
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
            # The only hard prerequisite is a working NVIDIA driver: the
            # PyTorch/TensorFlow wheels bundle the CUDA runtime and cuDNN,
            # and nvcc is only needed to COMPILE CUDA code - not to run the
            # frameworks. (The old system-cuDNN and nvcc checks probed
            # Linux-only paths, always failing on Windows/WSL and producing
            # false "prerequisites missing" warnings.)
            try:
                result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=5)
                if result.returncode != 0 and not Path('/usr/lib/wsl/lib/nvidia-smi').exists():
                    missing.append("NVIDIA drivers (nvidia-smi failed)")
            except subprocess.TimeoutExpired:
                missing.append("nvidia-smi timeout - driver issue?")
            except OSError:
                # Not found or not executable (e.g. a Windows binary resolved
                # via WSL PATH interop); WSL often only exposes it via
                # /usr/lib/wsl/lib (not on PATH)
                if not Path('/usr/lib/wsl/lib/nvidia-smi').exists():
                    missing.append("nvidia-smi not found - NVIDIA drivers not installed")

        elif gpu_type == 'rocm':
            # Only a working ROCm driver is required - the PyTorch ROCm
            # wheels bundle the ROCm runtime; hipcc is only for compiling
            # ROCm code, not for running the frameworks.
            try:
                result = subprocess.run(['rocm-smi'], capture_output=True, timeout=5)
                if result.returncode != 0:
                    missing.append("ROCm drivers (rocm-smi failed)")
            except subprocess.TimeoutExpired:
                missing.append("rocm-smi timeout - driver issue?")
            except OSError:
                missing.append("ROCm not installed (rocm-smi not found)")

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
    install_all: bool = False,
    probe_venv_path: Optional[str] = None,
    requirements_path: Optional[str] = None
) -> tuple[Path, HardwareInfo, str, list[tuple[str, str]], bool]:
    """Create venv configured for detected hardware (probe-then-commit).

    Args:
        venv_path: Target venv (created if missing, updated in place if present)
        output_config_path: Where to write hardware config JSON (optional)
        on_fail: Legacy option, accepted for compatibility (probe failures
            never touch the target venv)
        install_all: If True, install all working backends (not just priority)
        probe_venv_path: Disposable probe venv (e.g. .venv-probe) where
            candidates are tested before committing into venv_path; None
            selects legacy single-venv mode (candidates probed in target)
        requirements_path: Explicit path to the project's requirements.txt
            (None = auto-discover; see VenvBuilder._install_remaining_deps)

    Returns:
        (venv_path, hardware_info, keras_backend, all_successful_backends,
        verification_passed)
    """
    # Detect hardware
    detector = HardwareDetector()
    hardware_info = detector.detect()

    # Create venv with probe-then-commit
    builder = VenvBuilder(venv_path, hardware_info, on_fail=on_fail,
                          install_all=install_all, probe_venv_path=probe_venv_path,
                          requirements_path=requirements_path)
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

    # Run verification now so its result can be recorded in the config.
    # The return value is propagated to the caller (previously discarded,
    # which masked real failures behind a success summary).
    verify_passed = _verify_installation(venv)

    # Write hardware config with keras_backend
    if output_config_path:
        import json
        config_path = Path(output_config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = hardware_info.to_dict()
        config['keras_backend'] = keras_backend
        config['available_backends'] = [b[0] for b in successful_backends]
        config['verification_passed'] = verify_passed

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Hardware config written to {config_path}")
        logger.info(f"Primary Keras backend: {keras_backend}")
        if len(successful_backends) > 1:
            logger.info(f"Alternative backends available: {[b[0] for b in successful_backends[1:]]}")

    # Generate keras_backend.py for model-core with all available backends
    _generate_keras_backend_py(venv_path, keras_backend, successful_backends)

    return venv, hardware_info, keras_backend, successful_backends, verify_passed


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
        # Force UTF-8 for the child's stdout/stderr so its output never
        # crashes on Windows consoles (cp1252); decode it accordingly.
        child_env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        result = subprocess.run(
            [str(python_path), str(verify_script)],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=VERIFY_TIMEOUT,
            env=child_env
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
            logger.info("[OK] Verification PASSED")
        else:
            logger.warning("[FAIL] Verification FAILED")

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

    parser = argparse.ArgumentParser(description='Create ML venv for detected hardware '
                                                 '(probe-then-commit; target venv is never deleted)')
    parser.add_argument('venv_path', help='Target venv path (created if missing, updated in place if present)')
    parser.add_argument('--config', help='Path to write hardware config JSON')
    parser.add_argument('--on-fail', choices=['a', 'y', 'n'], default='n',
                        help='Legacy option, accepted for compatibility '
                             '(probe failures never touch the target venv)')
    parser.add_argument('--all', action='store_true', dest='install_all',
                        help='Install ALL working backends with commented switch options')
    parser.add_argument('--probe-venv', dest='probe_venv', default=None,
                        help='Disposable probe venv where candidates are tested '
                             '(default: no probe - candidates probed in target venv)')
    parser.add_argument('--requirements', dest='requirements', default=None,
                        help='Path to requirements.txt for the target venv '
                             '(default: auto-discover)')
    args = parser.parse_args()

    venv, hardware, keras_backend, all_backends, verify_ok = create_venv_for_hardware(
        args.venv_path, args.config, args.on_fail, args.install_all,
        probe_venv_path=args.probe_venv, requirements_path=args.requirements
    )

    if verify_ok:
        print(f"\n[OK] Virtual environment created at: {venv}")
    else:
        print(f"\n[!!] Virtual environment created at: {venv} - verification FAILED")
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

    # Non-zero exit code when verification failed so scripts/CI can detect it
    sys.exit(0 if verify_ok else 1)
