#!/usr/bin/env python3
"""Model-setup bootstrap wrapper (probe-then-commit venv flow).

Run from a project root (e.g. model-core) with the system Python:

    python3 test_venv_builder.py venv --config hardware_config.json

All created paths (target venv, .venv-probe, config, log) are anchored
to the CURRENT directory - the folder the script is invoked from -
never to the folder where this script file happens to live.

Bootstrap phase (system Python):
  - Sweeps any stale probe venv (.venv-probe in the current directory)
  - Creates a fresh disposable probe venv
  - Installs model-setup into it: from a locally built wheel via
    --model-setup-wheel, from the local source tree when this script
    lives inside the model-setup repository, or from PyPI otherwise
  - Re-executes itself from the probe venv

Probe phase (probe venv Python):
  - model-setup probes framework candidates INSIDE the probe venv
  - Only the proven package set is committed into the target venv
  - The probe venv is removed at the end of the run

The target venv is never deleted: it is created if missing and updated
in place if it already exists.
"""
import argparse
import logging
import os
import platform
import shutil
import subprocess
import sys
import venv
from datetime import datetime
from pathlib import Path

# Harden every child process (and the re-exec'd probe phase) against console
# encoding crashes (Windows cp1252 cannot encode all Unicode) and silence
# pip's self-upgrade notices. The re-exec'd interpreter picks these up at
# startup through the inherited environment.
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PIP_DISABLE_PIP_VERSION_CHECK', '1')

# All created paths anchor to the current directory, not the script location
PROBE_VENV = Path.cwd() / '.venv-probe'
# First PyPI release implementing the probe-then-commit flow
MINIMUM_MODEL_SETUP = 'model-setup>=0.3.0'


def _is_in_venv() -> bool:
    """Check if currently running inside a virtual environment."""
    return (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None
    )


def _running_from_probe() -> bool:
    """True when this process already runs from the probe venv."""
    try:
        return _is_in_venv() and Path(sys.prefix).resolve() == PROBE_VENV.resolve()
    except OSError:
        return False


def _remove_probe_venv(reason: str):
    """Best-effort removal of the probe venv.

    Only a process RUNNING from the probe venv cannot delete it on Windows
    (its own python.exe is locked); that one case schedules a detached,
    delayed deletion that completes a few seconds after this process exits.
    Every other caller (e.g. the stale sweep from the system interpreter)
    deletes the probe venv immediately on all platforms - this fixes probe
    venvs accumulating gigabytes of stale packages between runs.
    """
    if not PROBE_VENV.exists():
        return
    if _running_from_probe() and platform.system() == 'Windows':
        print(f"Note: probe venv still in use - scheduling delayed deletion ({reason})")
        subprocess.Popen(
            f'cmd /c timeout /t 3 /nobreak >nul & rmdir /s /q "{PROBE_VENV}"',
            shell=True,
        )
        return
    shutil.rmtree(PROBE_VENV, ignore_errors=True)
    print(f"Removed probe venv ({reason}): {PROBE_VENV}")


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description='Create ML venv for detected hardware '
                    '(probe-then-commit; the target venv is never deleted)'
    )
    parser.add_argument(
        'venv_path',
        help='Path to target venv (created if missing, updated in place if present)'
    )
    parser.add_argument('--config', help='Path to write hardware config JSON')
    parser.add_argument('--on-fail', choices=['a', 'y', 'n'], default='n',
                        help='Legacy option, accepted for compatibility '
                             '(probe failures never touch the target venv)')
    parser.add_argument('--all', action='store_true', dest='install_all',
                        help='Install ALL working backends with commented switch options')
    parser.add_argument('--log-file',
                        help='Path to log file (default: venv_path/../test_venv_builder.log)')
    parser.add_argument('--model-setup-wheel', dest='model_setup_wheel', default=None,
                        help='Pre-publish validation: install model-setup from this '
                             'wheel instead of PyPI / local source')
    parser.add_argument('--requirements', dest='requirements', default=None,
                        help='Path to requirements.txt for the target venv '
                             '(default: auto-discover)')
    return parser.parse_args(argv)


def _bootstrap(args):
    """Phase 0: fresh probe venv with model-setup, then re-exec from it."""
    _remove_probe_venv('stale sweep')
    print(f"Creating probe venv: {PROBE_VENV}")
    venv.create(str(PROBE_VENV), with_pip=True)

    if platform.system() == 'Windows':
        pip = PROBE_VENV / 'Scripts' / 'pip.exe'
        python = PROBE_VENV / 'Scripts' / 'python.exe'
    else:
        pip = PROBE_VENV / 'bin' / 'pip'
        python = PROBE_VENV / 'bin' / 'python'

    cwd = Path.cwd()
    if args.model_setup_wheel:
        print(f"Installing model-setup from wheel: {args.model_setup_wheel}")
        cmd = [str(pip), 'install', str(args.model_setup_wheel)]
    elif ((cwd / 'pyproject.toml').is_file() and
          (cwd / 'src' / 'model_setup').is_dir()):
        # Current directory is a model-setup repository checkout: test
        # the local source tree instead of the PyPI release
        print(f"Installing model-setup from local source tree: {cwd}")
        cmd = [str(pip), 'install', str(cwd)]
    else:
        print(f"Installing model-setup from PyPI ({MINIMUM_MODEL_SETUP})")
        cmd = [str(pip), 'install', MINIMUM_MODEL_SETUP]
    subprocess.run(cmd, check=True)

    print(f"Restarting with probe venv Python: {python}")
    sys.stdout.flush()  # os.execv does not flush Python buffers
    os.execv(str(python), [str(python), str(Path(__file__).resolve())] + sys.argv[1:])


def _run(args):
    """Probe phase: build the target venv via probe-then-commit."""
    from model_setup import __version__
    from model_setup.venv_builder import create_venv_for_hardware

    venv_path = Path(args.venv_path)
    if venv_path.resolve() == PROBE_VENV.resolve():
        raise SystemExit('Target venv path must not be the probe venv (.venv-probe)')

    log_file = args.log_file or str(venv_path.parent / 'test_venv_builder.log')
    # If log directory does not exist, fall back to the current directory
    # (do not auto-create destination folders)
    if not Path(log_file).parent.exists():
        log_file = str(Path.cwd() / Path(log_file).name)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("Model-Setup CLI Started (probe-then-commit)")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Version: {__version__}")
    try:
        # Branch of the project being built (current directory)
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logger.info(f"Branch: {result.stdout.strip()}")
    except Exception:
        pass
    logger.info("=" * 60)

    target, hardware, keras_backend, all_backends, verify_ok = create_venv_for_hardware(
        args.venv_path, args.config, args.on_fail, args.install_all,
        probe_venv_path=str(PROBE_VENV), requirements_path=args.requirements
    )

    if verify_ok:
        print(f"\n[OK] Virtual environment ready at: {target}")
    else:
        print(f"\n[!!] Virtual environment ready at: {target} - verification FAILED")
        print(f"     Check the log for details: {log_file}")
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
        print(f"  {target}\\Scripts\\activate")
    else:
        print(f"  source {target}/bin/activate")

    # Non-zero exit code when verification failed so scripts/CI can detect it
    sys.exit(0 if verify_ok else 1)


if __name__ == '__main__':
    parsed_args = _parse_args(sys.argv[1:])
    if _running_from_probe():
        try:
            _run(parsed_args)
        finally:
            _remove_probe_venv('run finished')
    else:
        _bootstrap(parsed_args)
