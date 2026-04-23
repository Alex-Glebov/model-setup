"""Check package versions available on PyPI for current platform.

Used to determine which backends can be installed before attempting.
"""

import json
import logging
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from .version_requirements import BACKEND_REQUIREMENTS, compare_versions

logger = logging.getLogger(__name__)


def get_pip_available_versions(package: str) -> List[str]:
    """Get list of available versions from pip index.

    Args:
        package: Package name

    Returns:
        List of available versions
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", package],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            logger.debug(f"pip index versions failed for {package}: {result.stderr}")
            return []

        # Parse output: "Available versions: 3.12.1, 3.12.0, ..."
        for line in result.stdout.split("\n"):
            if "Available versions:" in line:
                versions_str = line.split(":")[1]
                return [v.strip() for v in versions_str.split(",")]

        return []

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"Error checking {package}: {e}")
        return []


def is_version_available(package: str, min_version: str) -> Tuple[bool, Optional[str]]:
    """Check if a minimum version is available on PyPI.

    Args:
        package: Package name
        min_version: Minimum required version

    Returns:
        (is_available, latest_version or None)
    """
    versions = get_pip_available_versions(package)

    if not versions:
        return False, None

    latest = versions[0] if versions else None

    # Check if any version meets the minimum
    for version in versions:
        if compare_versions(version, min_version):
            return True, latest

    return False, latest


def check_backend_pip_availability(backend: str) -> Dict[str, Tuple[bool, Optional[str]]]:
    """Check which packages for a backend are available on PyPI.

    Args:
        backend: 'torch', 'tensorflow', or 'jax'

    Returns:
        {package: (is_available, latest_version)}
    """
    requirements = BACKEND_REQUIREMENTS.get(backend, [])
    results = {}

    for req in requirements:
        if req.name == "python":
            # Python version comes from system, not pip
            py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            results[req.name] = (True, py_version)
            continue

        available, latest = is_version_available(req.name, req.min_version)
        results[req.name] = (available, latest)

        if not available:
            logger.warning(
                f"Package {req.name}>={req.min_version} not available on PyPI "
                f"(latest: {latest or 'unknown'})"
            )

    return results


def can_install_backend(backend: str) -> Tuple[bool, Dict[str, Tuple[bool, Optional[str]]]]:
    """Check if a backend can be fully installed from PyPI.

    Args:
        backend: 'torch', 'tensorflow', or 'jax'

    Returns:
        (can_install, {package: (is_available, latest_version)})
    """
    results = check_backend_pip_availability(backend)
    can_install = all(available for available, _ in results.values())
    return can_install, results


def get_installable_backends() -> List[str]:
    """Get list of backends that can be installed from PyPI.

    Returns:
        List of backend names
    """
    installable = []

    for backend in ["torch", "tensorflow", "jax"]:
        can_install, details = can_install_backend(backend)
        if can_install:
            installable.append(backend)
            logger.info(f"Backend '{backend}' can be installed from PyPI")
        else:
            missing = [pkg for pkg, (available, _) in details.items() if not available]
            logger.debug(f"Backend '{backend}' missing packages on PyPI: {missing}")

    return installable


if __name__ == "__main__":
    # CLI usage
    logging.basicConfig(level=logging.INFO)

    print("Checking PyPI package availability...\n")
    print(f"Platform: {sys.platform} {sys.version_info.machine if hasattr(sys.version_info, 'machine') else 'unknown'}")
    print(f"Python: {sys.version}\n")

    for backend in ["torch", "tensorflow", "jax"]:
        can_install, details = can_install_backend(backend)
        status = "✓ CAN INSTALL" if can_install else "✗ CANNOT INSTALL"
        print(f"Backend: {backend} {status}")

        for pkg, (available, latest) in details.items():
            symbol = "✓" if available else "✗"
            req = BACKEND_REQUIREMENTS.get(backend, [])
            min_ver = next((r.min_version for r in req if r.name == pkg), "?")
            print(f"  {symbol} {pkg}: need >={min_ver}, latest={latest or 'unknown'}")
        print()

    installable = get_installable_backends()
    print(f"\nInstallable backends: {installable}")
