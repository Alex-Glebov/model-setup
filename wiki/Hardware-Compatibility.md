# Hardware Compatibility

Complete compatibility matrix for model-setup.

## Supported Platforms

### NVIDIA Jetson

| Device | JetPack | PyTorch | Status |
|--------|---------|---------|--------|
| Orin Nano | 6.0+ | 2.5.0 | ✅ Supported |
| Orin NX | 6.0+ | 2.5.0 | ✅ Supported |
| Orin AGX | 6.0+ | 2.5.0 | ✅ Supported |
| Xavier NX | 5.1 | 2.1.0 | ⚠️ Not tested |
| Xavier AGX | 5.1 | 2.1.0 | ⚠️ Not tested |

**Requirements**:
- JetPack 6.x (R36.x in `/etc/nv_tegra_release`)
- Python 3.10
- 4GB+ RAM

**Compute Capability**: sm_87 (Orin)

### NVIDIA CUDA

| CUDA | PyTorch | Status |
|------|---------|--------|
| 12.1 | 2.5.0 | ✅ Supported |
| 11.8 | 2.5.0 | ✅ Supported |
| 11.7 | 2.0.0 | ✅ Supported |

**Requirements**:
- NVIDIA driver 525.60.13+
- CUDA toolkit installed
- x86_64 or aarch64

### AMD ROCm

| ROCm | PyTorch | Status |
|------|---------|--------|
| 5.7 | 2.5.0 | ✅ Supported |
| 5.6 | 2.0.0 | ✅ Supported |

**Requirements**:
- ROCm 5.7+ installed
- AMD GPU (Radeon or Instinct)
- x86_64

### CPU Only

| Platform | PyTorch | Status |
|----------|---------|--------|
| Linux x86_64 | 2.5.0 | ✅ Supported |
| Linux aarch64 | 2.5.0 | ✅ Supported |

## Feature Comparison

| Feature | Jetson | CUDA | ROCm | CPU |
|---------|--------|------|------|-----|
| PyTorch GPU | ✅ | ✅ | ✅ | ❌ |
| TensorFlow GPU | ⚠️ | ✅ | ✅ | ❌ |
| cuSPARSELt | ✅ | ❌ | ❌ | ❌ |
| Mixed Precision | ✅ | ✅ | ✅ | ❌ |
| Multi-GPU | ❌ | ❌ | ❌ | ❌ |

*Note: TensorFlow GPU not officially supported on JetPack 6.x*

## Performance Benchmarks

### Jetson Orin Nano

| Configuration | Epoch Time | Memory |
|---------------|------------|--------|
| Proof-of-concept (32 units, batch 16) | ~1.2s | ~2GB |
| Standard (128 units, batch 32) | ~3.5s | ~4GB |
| Full (256 units, batch 64) | ~7s | ~6GB |

### Desktop CUDA (RTX 4090)

| Configuration | Epoch Time | Memory |
|---------------|------------|--------|
| Standard (128 units, batch 64) | ~0.5s | ~6GB |
| Full (256 units, batch 128) | ~1s | ~10GB |

## Known Limitations

### Jetson

- **TensorFlow**: Not officially supported on JetPack 6.x
- **torchvision**: Must be built from source (no wheel available)
- **Memory**: 8GB shared memory (RAM + GPU)
- **Swap**: Recommended for training large models

### CUDA

- **cuDNN**: Must match PyTorch compilation version
- **Driver**: Must be compatible with CUDA version

### ROCm

- **torchvision**: Limited support
- **Windows**: Not supported (Linux only)

### CPU

- **Speed**: 10-20x slower than GPU
- **Large models**: May run out of memory

## Future Support

Planned additions:

- [ ] Jetson Nano (2GB/4GB)
- [ ] NVIDIA RTX 5050
- [ ] AMD Ryzen AI
- [ ] Intel Arc GPUs
- [ ] Apple Silicon (MPS)

## Testing Your Hardware

```python
# Test script
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test tensor operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print(f"Tensor test: OK")
```

## Reporting Compatibility

If your hardware works but isn't listed:

1. Run the test script above
2. Note your hardware specs
3. Open an issue with results
