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
    args = parser.parse_args()

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
    try:
        from model_setup import __version__
        logger.info(f"Version: {__version__}")
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


if __name__ == '__main__':
    main()
