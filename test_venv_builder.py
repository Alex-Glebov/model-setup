#!/usr/bin/env python3
"""Model-setup CLI entry point.

Tries to import model_setup. If unavailable:
  - Checks if running inside a virtual environment
  - If not, creates one in the same directory as this script
  - Installs model_setup from the local src/ or PyPI
  - Re-executes with the venv's Python

After import succeeds, delegates to create_venv_for_hardware().
"""
import platform
import subprocess
import sys
import os
import venv
from pathlib import Path


def _is_in_venv() -> bool:
    """Check if currently running inside a virtual environment."""
    return (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None
    )


def _ensure_model_setup():
    """Ensure model_setup is importable.

    1. Try direct import.
    2. If in a venv, install into it.
    3. If not in a venv, create one next to this script and install there.
    4. Re-exec with the venv Python so imports work.
    """
    try:
        import model_setup
        return  # Already available
    except ImportError:
        pass

    script_dir = Path(__file__).resolve().parent

    if _is_in_venv():
        # We're already in a venv — just install the package here
        pip = Path(sys.executable).parent / 'pip'
        if not pip.exists():
            pip = Path(sys.executable).parent / 'pip3'
        print("model_setup not found. Installing into current venv...")
        subprocess.run([str(pip), 'install', '--no-deps', '-e', str(script_dir)], check=True)
        return  # Will import successfully on next try

    # Not in a venv — create one next to this script
    venv_path = script_dir / '.venv'
    print(f"model_setup not found. Creating venv at {venv_path} ...")
    venv.create(str(venv_path), with_pip=True)

    # Determine pip path
    if platform.system() == 'Windows':
        pip = venv_path / 'Scripts' / 'pip.exe'
        python = venv_path / 'Scripts' / 'python.exe'
    else:
        pip = venv_path / 'bin' / 'pip'
        python = venv_path / 'bin' / 'python'

    # Install model_setup from local root (pyproject.toml lives there)
    print(f"Installing model_setup into venv...")
    subprocess.run([str(pip), 'install', '--no-deps', '-e', str(script_dir)], check=True)

    # Re-exec with venv Python, passing all original args
    print(f"Restarting with venv Python: {python}")
    os.execv(str(python), [str(python), __file__] + sys.argv[1:])


if __name__ == '__main__':
    _ensure_model_setup()

    import argparse
    import logging
    from datetime import datetime

    from model_setup import __version__
    from model_setup.venv_builder import create_venv_for_hardware

    parser = argparse.ArgumentParser(description='Create ML venv for detected hardware')
    parser.add_argument('venv_path', help='Path to create venv')
    parser.add_argument('--config', help='Path to write hardware config JSON')
    parser.add_argument('--on-fail', choices=['a', 'y', 'n'], default='n',
                        help='Action on failed install: a=auto-delete, y=ask, n=keep (default)')
    parser.add_argument('--all', action='store_true', dest='install_all',
                        help='Install ALL working backends with commented switch options')
    parser.add_argument('--log-file', help='Path to log file (default: venv_path/../test_venv_builder.log)')
    args = parser.parse_args()

    # Setup logging to both console and file
    venv_path = Path(args.venv_path)
    log_file = args.log_file or str(venv_path.parent / 'test_venv_builder.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Model-Setup CLI Started")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Version: {__version__}")
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent)
        )
        if result.returncode == 0:
            logger.info(f"Branch: {result.stdout.strip()}")
    except Exception:
        pass
    logger.info("=" * 60)

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
