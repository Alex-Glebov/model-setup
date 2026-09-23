"""Allow running model_setup as a module: python -m model_setup."""
import sys


def main():
    """Entry point for console script."""
    from model_setup.venv_builder import create_venv_for_hardware

    import argparse
    import logging
    import platform
    from pathlib import Path
    from datetime import datetime

    parser = argparse.ArgumentParser(description='Create ML venv for detected hardware')
    parser.add_argument('venv_path', help='Path to create venv')
    parser.add_argument('--config', help='Path to write hardware config JSON')
    parser.add_argument('--on-fail', choices=['a', 'y', 'n'], default='n',
                        help='Action on failed install: a=auto-delete, y=ask, n=keep (default)')
    parser.add_argument('--all', action='store_true', dest='install_all',
                        help='Install ALL working backends with commented switch options')
    parser.add_argument('--log-file', help='Path to log file (default: venv_path/../test_venv_builder.log)')
    parser.add_argument('--probe-venv', dest='probe_venv', default=None,
                        help='Disposable probe venv where candidates are tested '
                             '(default: no probe - candidates probed in target venv)')
    parser.add_argument('--requirements', dest='requirements', default=None,
                        help='Path to requirements.txt for the target venv '
                             '(default: auto-discover)')
    args = parser.parse_args()

    venv_path = Path(args.venv_path)
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
    logger.info("Model-Setup CLI Started")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Log file: {log_file}")
    try:
        from model_setup import __version__
        logger.info(f"Version: {__version__}")
    except Exception:
        pass
    logger.info("=" * 60)

    venv, hardware, keras_backend, all_backends, verify_ok = create_venv_for_hardware(
        args.venv_path, args.config, args.on_fail, args.install_all,
        probe_venv_path=args.probe_venv, requirements_path=args.requirements
    )

    if verify_ok:
        print(f"\n[OK] Virtual environment created at: {venv}")
    else:
        print(f"\n[!!] Virtual environment created at: {venv} - verification FAILED")
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
        print(f"  {venv}\\Scripts\\activate")
    else:
        print(f"  source {venv}/bin/activate")

    # Non-zero exit code when verification failed so scripts/CI can detect it
    sys.exit(0 if verify_ok else 1)


if __name__ == '__main__':
    main()
