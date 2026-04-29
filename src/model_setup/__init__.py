"""Model Setup - Hardware detection and environment setup for ML training.

This package handles:
1. Hardware identification (GPU type, capabilities)
2. Driver and dependency installation
3. Virtual environment creation
4. Configuration file generation for model-core
"""

__version__ = "0.2.2"

# Default timeouts (seconds)
TF_TEST_TIMEOUT = 120      # TensorFlow first import probes GPUs, compiles kernels
UNINSTALL_TIMEOUT = 120  # Backend uninstall can be slow due to many packages
