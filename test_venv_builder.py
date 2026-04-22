#!/usr/bin/env python3
"""Test script to run venv builder directly."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from model_setup.venv_builder import create_venv_for_hardware

if __name__ == '__main__':
    import argparse
    import logging

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Create ML venv for detected hardware')
    parser.add_argument('venv_path', help='Path to create venv')
    parser.add_argument('--config', help='Path to write hardware config JSON')
    args = parser.parse_args()

    venv, hardware = create_venv_for_hardware(args.venv_path, args.config)

    print(f"\n✓ Virtual environment created at: {venv}")
    print(f"  Hardware: {hardware.gpu_type or 'CPU-only'}")
    print(f"  GPU: {hardware.gpu_name or 'N/A'}")
    print(f"\nTo activate:")
    print(f"  source {venv}/bin/activate")
