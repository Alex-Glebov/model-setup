"""Minimum version requirements for ML training dependencies.

Used to validate available packages before building venv.
All versions must be satisfied for a backend to be considered available.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PackageRequirement:
    """Package requirement with version constraints."""
    name: str
    min_version: str
    import_name: Optional[str] = None  # Different from pip name, e.g., 'sklearn' vs 'scikit-learn'


# Minimum supported versions for Keras 3.x backends
# See: https://keras.io/getting_started/
KERAS_3_MINIMUM_VERSIONS = {
    "python": "3.9.0",
    "keras": "3.0.0",
    "torch": "2.1.0",
    "tensorflow": "2.15.0",
    "jax": "0.4.20",
    "jaxlib": "0.4.20",
}

# Package import name mappings (pip name -> import name)
IMPORT_NAME_MAP = {
    "scikit-learn": "sklearn",
    "tensorflow-cpu": "tensorflow",
    "tensorflow-gpu": "tensorflow",
    "torch": "torch",
    "keras": "keras",
}


def get_import_name(package: str) -> str:
    """Get the import name for a package.

    Args:
        package: pip package name

    Returns:
        Import name to use in __import__
    """
    return IMPORT_NAME_MAP.get(package, package)


def compare_versions(installed: str, required: str) -> bool:
    """Compare version strings.

    Args:
        installed: Installed version (e.g., "2.5.0")
        required: Required version (e.g., "2.1.0")

    Returns:
        True if installed >= required
    """
    def normalize(v: str) -> List[int]:
        """Convert version string to comparable list."""
        # Handle versions like "2.5.0a0+872d972e41"
        v = v.split('+')[0]  # Remove build metadata
        v = v.replace('a', '.')  # Replace alpha markers with dots
        v = v.replace('b', '.')  # Replace beta markers
        v = v.replace('rc', '.')  # Replace RC markers
        return [int(x) for x in v.split('.') if x.isdigit()]

    try:
        installed_parts = normalize(installed)
        required_parts = normalize(required)

        # Pad shorter version with zeros
        max_len = max(len(installed_parts), len(required_parts))
        installed_parts.extend([0] * (max_len - len(installed_parts)))
        required_parts.extend([0] * (max_len - len(required_parts)))

        return installed_parts >= required_parts
    except (ValueError, AttributeError):
        return False


# Backend requirements
BACKEND_REQUIREMENTS: Dict[str, List[PackageRequirement]] = {
    "torch": [
        PackageRequirement("python", "3.9.0"),
        PackageRequirement("keras", "3.0.0"),
        PackageRequirement("torch", "2.1.0"),
    ],
    "tensorflow": [
        PackageRequirement("python", "3.9.0"),
        PackageRequirement("keras", "3.0.0"),
        PackageRequirement("tensorflow", "2.15.0"),
    ],
    "jax": [
        PackageRequirement("python", "3.9.0"),
        PackageRequirement("keras", "3.0.0"),
        PackageRequirement("jax", "0.4.20"),
        PackageRequirement("jaxlib", "0.4.20"),
    ],
}


def get_backend_requirements(backend: str) -> List[PackageRequirement]:
    """Get package requirements for a backend.

    Args:
        backend: 'torch', 'tensorflow', or 'jax'

    Returns:
        List of package requirements
    """
    return BACKEND_REQUIREMENTS.get(backend, [])
