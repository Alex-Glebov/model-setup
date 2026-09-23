#!/usr/bin/env python3
"""Verification script for model-setup installation.

Checks that installed backends are working correctly.
Run directly with venv Python: /path/to/venv/bin/python verify.py

Output uses ASCII markers ([OK]/[FAIL]/[--]) so it is encodable on every
console, including Windows legacy code pages (cp1252) that cannot encode
characters like U+2713 and would otherwise crash the verification run.
"""

import sys
import os


def _ensure_unicode_safe_output():
    """Make stdout/stderr survive any console encoding.

    ASCII markers keep this script's own output safe everywhere; this
    reconfigure additionally protects error messages that may contain
    arbitrary Unicode produced by imported libraries (e.g. a torch
    import traceback), which would otherwise abort verification midway
    and report a false failure on cp1252 consoles.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            try:
                stream.reconfigure(encoding='utf-8', errors='replace')
            except Exception:
                pass


def verify_torch():
    """Verify PyTorch installation."""
    try:
        import torch
        print(f"[OK] PyTorch: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"  CUDA available: True")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**2)} MB")
            return True
        else:
            print("  CUDA available: False (CPU mode)")
            return True

    except ImportError:
        print("[--] PyTorch: Not installed")
        return None
    except Exception as e:
        print(f"[FAIL] PyTorch: Error - {e}")
        return False


def verify_tensorflow():
    """Verify TensorFlow installation."""
    try:
        import tensorflow as tf
        print(f"[OK] TensorFlow: {tf.__version__}")

        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  GPU available: True")
            for gpu in gpus:
                print(f"    {gpu}")
        else:
            print("  GPU available: False (CPU mode)")

        return True

    except ImportError:
        print("[--] TensorFlow: Not installed")
        return None
    except Exception as e:
        print(f"[FAIL] TensorFlow: Error - {e}")
        return False


def verify_keras():
    """Verify Keras installation and backend."""
    try:
        # Set default backend if not set (required for Keras 3.x)
        if not os.environ.get('KERAS_BACKEND'):
            os.environ['KERAS_BACKEND'] = 'torch'

        import keras

        # Try to get backend info
        backend_name = None
        if hasattr(keras, 'config') and hasattr(keras.config, 'backend'):
            backend_name = keras.config.backend()
        elif hasattr(keras, 'backend'):
            backend_name = str(keras.backend())

        if backend_name:
            print(f"[OK] Keras: {keras.__version__} (backend: {backend_name})")
        else:
            print(f"[OK] Keras: {keras.__version__}")

        return True

    except ImportError:
        print("[--] Keras: Not installed")
        return None
    except Exception as e:
        print(f"[FAIL] Keras: Error - {e}")
        return False


def main():
    """Run verification checks."""
    _ensure_unicode_safe_output()

    print("=" * 50)
    print("Model-Setup Installation Verification")
    print("=" * 50)
    print()

    results = {}

    # Check each component
    print("Checking PyTorch...")
    results['torch'] = verify_torch()
    print()

    print("Checking TensorFlow...")
    results['tensorflow'] = verify_tensorflow()
    print()

    print("Checking Keras...")
    results['keras'] = verify_keras()
    print()

    # Summary
    print("=" * 50)
    print("Summary")
    print("=" * 50)

    working = [k for k, v in results.items() if v]
    not_installed = [k for k, v in results.items() if v is None]
    failed = [k for k, v in results.items() if v is False]

    if working:
        print(f"[OK] Working: {', '.join(working)}")
    if not_installed:
        print(f"[--] Not installed: {', '.join(not_installed)}")
    if failed:
        print(f"[FAIL] Failed: {', '.join(failed)}")

    print()

    # Return exit code - only fail if something actually failed (not just not installed)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
