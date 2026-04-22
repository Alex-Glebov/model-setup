# Contributing to Model Setup

Thank you for your interest in contributing to model-setup!

## Development Setup

### Prerequisites

- Python 3.10+
- Linux (aarch64 for Jetson testing, x86_64 for CUDA/ROCm)
- Git

### Clone and Setup

```bash
git clone https://github.com/Alex-Glebov/model-setup.git
cd model-setup

# Optional: Create virtual environment for development
python3 -m venv dev_venv
source dev_venv/bin/activate
pip install -e .
```

## Project Structure

```
model-setup/
├── src/model_setup/           # Main package
│   ├── hardware_detector.py   # GPU detection logic
│   └── venv_builder.py        # Venv creation
├── tests/                     # Test suite (to be added)
├── docs/                      # Documentation
└── scripts/                   # Utility scripts
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints where possible
- Maximum line length: 100 characters
- Docstrings in Google style

Example:
```python
def detect_gpu() -> Optional[str]:
    """Detect GPU type from system.

    Returns:
        GPU type ('jetson', 'cuda', 'rocm') or None
    """
    # Implementation
```

### Commit Messages

Format: `<type>: <subject>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat: add ROCm GPU detection
fix: handle missing nv_tegra_release file
docs: update Jetson compatibility matrix
```

## Testing

### Hardware Detection Tests

```bash
# Test on different platforms
python -c "from model_setup.hardware_detector import HardwareDetector; \
    print(HardwareDetector().detect())"
```

### Venv Creation Tests

```bash
# Create test venv (use /tmp for cleanup)
python test_venv_builder.py /tmp/test_venv --config /tmp/test_config.json

# Verify installation
/tmp/test_venv/bin/python -c "import torch; print(torch.__version__)"
```

## Adding New Hardware Support

To add support for a new GPU type:

1. **Update `hardware_detector.py`**:
   - Add detection logic in `detect()` method
   - Return new GPU type string

2. **Update `venv_builder.py`**:
   - Add `_configure_<new_gpu>()` method
   - Update `_apply_hardware_config()` to call new method
   - Install appropriate PyTorch wheel

3. **Update documentation**:
   - Add to README.md hardware support table
   - Update Wiki with setup instructions

4. **Test**:
   - Test on actual hardware
   - Verify PyTorch CUDA/GPU availability

## Reporting Issues

When reporting issues, please include:

1. **Hardware**: GPU type, JetPack version (if Jetson), OS
2. **Python version**: `python --version`
3. **Error message**: Full traceback if applicable
4. **Steps to reproduce**: Minimal code example

Example:
```
Hardware: Jetson Orin Nano, JetPack 6.0
Python: 3.10.12
Error: ImportError: libcusparseLt.so.0: cannot open shared object file
Steps: Ran `python test_venv_builder.py venv --config config.json`, then `source venv/bin/activate; python -c "import torch"`
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes with tests
4. Commit with clear messages
5. Push to your fork
6. Open PR against `develop` branch

## Code Review

All PRs require:
- At least one review
- CI passing (if applicable)
- No merge conflicts

## Questions?

- Open an issue for questions
- Check existing issues and Wiki first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
