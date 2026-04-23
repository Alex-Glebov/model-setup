"""Package version checker for ML dependencies.

Validates that installed packages meet minimum version requirements.
"""

import importlib
import logging
import sys
from typing import Dict, List, Optional, Tuple

from .version_requirements import (
    BACKEND_REQUIREMENTS,
    compare_versions,
    get_import_name,
    PackageRequirement,
)

logger = logging.getLogger(__name__)


def get_package_version(package: str) -> Optional[str]:
    """Get installed version of a package.

    Args:
        package: Package name (pip name)

    Returns:
        Version string or None if not installed
    """
    import_name = get_import_name(package)

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", None)
        if version:
            return version

        # Fallback: try importlib.metadata
        try:
            from importlib.metadata import version as get_version
            return get_version(package)
        except ImportError:
            pass

        return None
    except ImportError:
        return None


def check_requirement(req: PackageRequirement) -> Tuple[bool, Optional[str]]:
    """Check if a single requirement is satisfied.

    Args:
        req: Package requirement

    Returns:
        (is_satisfied, installed_version or None)
    """
    if req.name == "python":
        # Check Python version
        current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return compare_versions(current, req.min_version), current

    installed = get_package_version(req.name)
    if installed is None:
        return False, None

    return compare_versions(installed, req.min_version), installed


def check_backend_availability(backend: str) -> Tuple[bool, Dict[str, Tuple[bool, Optional[str]]]]:
    """Check if a backend is available with required versions.

    Args:
        backend: 'torch', 'tensorflow', or 'jax'

    Returns:
        (all_available, {package: (is_satisfied, installed_version)})
    """
    requirements = BACKEND_REQUIREMENTS.get(backend, [])
    if not requirements:
        return False, {}

    results = {}
    all_satisfied = True

    for req in requirements:
        satisfied, version = check_requirement(req)
        results[req.name] = (satisfied, version)
        if not satisfied:
            all_satisfied = False
            logger.warning(
                f"Backend '{backend}' requirement failed: {req.name} "
                f"(have {version or 'not installed'}, need >= {req.min_version})"
            )

    return all_satisfied, results


def get_available_backends() -> List[str]:
    """Get list of available backends on this system.

    Returns:
        List of backend names ('torch', 'tensorflow', 'jax') that are available
    """
    available = []

    for backend in ["torch", "tensorflow", "jax"]:
        is_available, details = check_backend_availability(backend)
        if is_available:
            available.append(backend)
            logger.info(f"Backend '{backend}' is available")
        else:
            missing = [pkg for pkg, (satisfied, _) in details.items() if not satisfied]
            logger.debug(f"Backend '{backend}' not available, missing: {missing}")

    return available


def format_backend_status(backend: str) -> str:
    """Format backend availability status for display.

    Args:
        backend: Backend name

    Returns:
        Formatted status string
    """
    is_available, details = check_backend_availability(backend)

    lines = [f"Backend: {backend}"]
    lines.append(f"  Available: {'Yes' if is_available else 'No'}")
    lines.append("  Requirements:")

    for pkg, (satisfied, version) in details.items():
        status = "✓" if satisfied else "✗"
        version_str = version or "not installed"
        req = BACKEND_REQUIREMENTS.get(backend, [])
        min_ver = next((r.min_version for r in req if r.name == pkg), "?")
        lines.append(f"    {status} {pkg}: {version_str} (need >= {min_ver})")

    return "\n".join(lines)


if __name__ == "__main__":
    # CLI usage for testing
    logging.basicConfig(level=logging.INFO)

    print("Checking available ML backends...\n")

    for backend in ["torch", "tensorflow", "jax"]:
        print(format_backend_status(backend))
        print()

    available = get_available_backends()
    print(f"Available backends: {available}")
