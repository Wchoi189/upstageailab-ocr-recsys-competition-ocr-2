---
ads_version: "1.0"
type: "guide"
experiment_id: "20251122_172313_perspective_correction"
status: "complete"
created: "2025-12-17T17:59:47Z"
updated: "2025-12-17T17:59:47Z"
tags: ['perspective-correction', 'setup', 'rembg', 'gpu']
commands: []
prerequisites: []
---

# rembg GPU Setup Status

**Date**: 2025-11-22
**Status**: ⚠️ GPU Detected, CUDA Libraries Need Configuration
**Author**: AI Agent

## Current Status

### ✅ GPU Hardware Detected
- **GPU**: NVIDIA GeForce RTX 3090
- **Driver Version**: 581.80
- **CUDA Version**: 13.0
- **GPU Memory**: 24GB (1.4GB used)

### ✅ PyTorch CUDA Available
- PyTorch can see and use the GPU
- CUDA device count: 1

### ✅ ONNX Runtime GPU Installed
- `onnxruntime-gpu==1.23.2` installed
- Providers detected: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

### ⚠️ CUDA Libraries Issue
- **Error**: `libcudnn.so.9: cannot open shared object file: No such file or directory`
- **Impact**: ONNX Runtime falls back to CPU execution
- **Status**: Session creation succeeds, but uses CPU provider

## What This Means

1. **GPU is properly detected** by the system and ONNX Runtime
2. **Test script will run** and test all configurations
3. **CPU baseline will work** (currently ~0.5-1.0s per image)
4. **GPU/TensorRT tests will show** the configuration but may still use CPU until CUDA libraries are fixed

## Fixing CUDA Library Issue

The missing `libcudnn.so.9` is part of the CUDA Deep Neural Network library. To fix:

### Option 1: Install CUDA Toolkit (Recommended)

```bash
# Check CUDA installation
nvcc --version

# If not installed, install CUDA toolkit matching your driver
# For CUDA 13.0, you may need to install from NVIDIA website
```

### Option 2: Set LD_LIBRARY_PATH

If CUDA is installed but not in library path:

```bash
# Find CUDA libraries
find /usr/local/cuda* -name "libcudnn.so*" 2>/dev/null

# Add to LD_LIBRARY_PATH (add to ~/.bashrc for persistence)
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Option 3: Install cuDNN

```bash
# Install cuDNN matching your CUDA version
# Download from: https://developer.nvidia.com/cudnn
# Or use package manager if available
```

## Current Test Results

Even with the CUDA library issue, the test script will:
- ✅ Test all configurations
- ✅ Show performance metrics
- ✅ Save output images
- ⚠️ Use CPU for all tests (until CUDA libraries fixed)

**Expected Performance (CPU)**:
- Baseline (silueta, 640px): ~0.5-1.0s per image
- After first image (session cached): ~0.5s per image

**Expected Performance (GPU - after fix)**:
- GPU (CUDA): ~0.1-0.2s per image (5-10x faster)
- TensorRT: ~0.05-0.1s per image (10-20x faster)

## Verification Commands

```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check ONNX Runtime providers
python -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Check optimized rembg detection
python -c "from scripts.optimized_rembg import GPU_AVAILABLE, TENSORRT_AVAILABLE; print(f'GPU: {GPU_AVAILABLE}, TensorRT: {TENSORRT_AVAILABLE}')"
```

## Next Steps

1. ✅ **GPU detection working** - Test script will detect GPU/TensorRT
2. ⚠️ **Fix CUDA libraries** - Install cuDNN or configure library paths
3. ✅ **Run tests** - Test script will work (using CPU until libraries fixed)
4. ⚠️ **Re-run after fix** - Get actual GPU performance metrics

## Notes

- The test script is designed to work even if GPU isn't fully functional
- All configurations will be tested
- Performance metrics will show CPU performance until CUDA libraries are fixed
- Once fixed, rembg will automatically use GPU when available

