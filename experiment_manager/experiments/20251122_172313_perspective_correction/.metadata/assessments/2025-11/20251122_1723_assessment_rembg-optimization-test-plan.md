---
ads_version: '1.0'
type: assessment
experiment_id: 20251122_172313_perspective_correction
status: complete
created: '2025-12-17T17:59:47Z'
updated: '2025-12-27T16:16:42.397554'
tags:
- perspective-correction
- testing
- optimization
- rembg
phase: phase_0
priority: medium
evidence_count: 0
title: 20251122 1723 Assessment Rembg-Optimization-Test-Plan
---
# rembg Optimization Test Plan

**Date**: 2025-01-30
**Status**: Ready for Testing
**Author**: AI Agent

## Overview

This document outlines the test plan for specific rembg optimization settings as requested:
- Model: `silueta` (fastest model)
- Image size: 640 (model training size)
- Alpha matting: Disabled
- GPU/TensorRT: Enabled for NVIDIA GPUs
- INT8 quantization: Tested if available

## Test Configurations

### Configuration 1: Baseline (CPU)
- **Model**: `silueta`
- **Image Size**: 640px
- **Alpha Matting**: Disabled
- **GPU**: No
- **TensorRT**: No
- **INT8**: No

**Expected**: Baseline performance for comparison

### Configuration 2: GPU (CUDA)
- **Model**: `silueta`
- **Image Size**: 640px
- **Alpha Matting**: Disabled
- **GPU**: Yes (CUDA)
- **TensorRT**: No
- **INT8**: No

**Expected**: 5-10x speedup over CPU

### Configuration 3: TensorRT
- **Model**: `silueta`
- **Image Size**: 640px
- **Alpha Matting**: Disabled
- **GPU**: Yes
- **TensorRT**: Yes
- **INT8**: No

**Expected**: Additional 1.5-2x speedup over CUDA

### Configuration 4: INT8 Quantization (If Available)
- **Model**: `silueta` (INT8 quantized)
- **Image Size**: 640px
- **Alpha Matting**: Disabled
- **GPU**: Yes (if available)
- **TensorRT**: Yes (if available)
- **INT8**: Yes

**Expected**: Additional 1.5-2x speedup, potential quality loss

**Note**: rembg doesn't officially provide INT8 quantized models. This configuration will test if custom quantized models are available or can be created.

## Test Scripts

### 1. Optimized rembg Test Script

**Location**: `scripts/test_optimized_rembg.py`

**Purpose**: Test all optimization configurations and compare performance

**Usage**:
```bash
python scripts/test_optimized_rembg.py \
    --input-dir data/samples \
    --output-dir outputs/optimized_test \
    --num-samples 10
```

**Features**:
- Tests all configurations automatically
- Compares performance metrics
- Saves outputs for each configuration
- Generates performance summary

### 2. Pipeline Test Script

**Location**: `scripts/test_pipeline_rembg_perspective.py`

**Purpose**: Test full pipeline (rembg → perspective correction) with optimized settings

**Usage**:
```bash
python scripts/test_pipeline_rembg_perspective.py \
    --input-dir data/samples \
    --output-dir outputs/pipeline_test \
    --num-samples 10
```

**Features**:
- Uses optimized rembg settings by default
- Applies perspective correction
- Generates comparison images

## Prerequisites

### 1. Install rembg with GPU Support

```bash
# Install rembg (already in pyproject.toml)
uv sync

# For GPU support, install onnxruntime-gpu
uv add onnxruntime-gpu

# Verify GPU support
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

**Expected output** (if GPU available):
```
['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 2. Install TensorRT (Optional, for NVIDIA GPUs)

TensorRT requires:
- NVIDIA GPU with CUDA support
- TensorRT library installed
- `onnxruntime-gpu` with TensorRT support

**Installation**:
```bash
# TensorRT is typically installed separately
# Check if available:
python -c "import onnxruntime as ort; print('TensorRT' in ort.get_available_providers())"
```

**Note**: TensorRT installation is complex and platform-specific. See [ONNX Runtime TensorRT documentation](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html).

### 3. INT8 Quantized Models

rembg doesn't officially provide INT8 quantized models. To use INT8:

1. **Option A**: Use custom quantized models (if available)
   - Check rembg model directory: `~/.u2net/`
   - Look for `*_int8.onnx` files

2. **Option B**: Quantize models yourself
   - Use ONNX Runtime quantization tools
   - Requires calibration dataset
   - See [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/quantization.html)

## Running Tests

### Step 1: Check System Capabilities

```bash
python scripts/test_optimized_rembg.py --input-dir data/samples --num-samples 1
```

This will show:
- rembg availability
- GPU availability
- TensorRT availability
- Available ONNX providers

### Step 2: Run Full Test Suite

```bash
python scripts/test_optimized_rembg.py \
    --input-dir data/samples \
    --output-dir outputs/optimized_test \
    --num-samples 10
```

### Step 3: Review Results

Check the output directory for:
- Processed images for each configuration
- Performance metrics in console output
- Comparison of different configurations

## Expected Performance

| Configuration | Expected Time | Speedup vs CPU |
|--------------|---------------|----------------|
| CPU (baseline) | ~1.0-2.0s | 1.0x |
| GPU (CUDA) | ~0.2-0.4s | 5-10x |
| TensorRT | ~0.1-0.2s | 10-20x |
| INT8 + TensorRT | ~0.05-0.1s | 20-40x |

*Note: Actual performance varies by image size, GPU model, and system configuration*

## Metrics to Collect

1. **Processing Time**: Time per image for each configuration
2. **Throughput**: Images per second
3. **Memory Usage**: GPU/CPU memory consumption
4. **Quality**: Visual comparison of outputs
5. **Success Rate**: Percentage of successful processing

## Troubleshooting

### GPU Not Detected

1. Check NVIDIA drivers:
   ```bash
   nvidia-smi
   ```

2. Check CUDA installation:
   ```bash
   nvcc --version
   ```

3. Verify onnxruntime-gpu:
   ```bash
   python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   ```

### TensorRT Not Available

1. TensorRT requires separate installation
2. Check ONNX Runtime TensorRT documentation
3. May require building from source

### INT8 Models Not Found

1. rembg doesn't provide INT8 models by default
2. Need to quantize models manually or use custom models
3. Check `~/.u2net/` directory for custom models

## Next Steps

1. ✅ Run baseline tests (CPU)
2. ✅ Test GPU acceleration (if available)
3. ✅ Test TensorRT (if available)
4. ⚠️ Test INT8 quantization (if models available)
5. ⚠️ Compare quality vs performance trade-offs
6. ⚠️ Document optimal configuration for production

## References

- [rembg GitHub](https://github.com/danielgatis/rembg)
- [ONNX Runtime GPU](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ONNX Runtime TensorRT](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/quantization.html)

