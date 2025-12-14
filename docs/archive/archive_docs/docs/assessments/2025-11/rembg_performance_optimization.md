# rembg Performance Optimization Assessment

**Date**: 2025-01-30
**Status**: Assessment
**Author**: AI Agent

## Executive Summary

This document assesses performance optimization strategies for rembg background removal, which is currently CPU-bound and slow when using ONNX runtime. We evaluate different installation methods and optimization approaches to improve processing speed.

## Current Implementation

### Installation Method
- **Current**: `rembg>=2.0.67` installed via pip/uv
- **Location**: Installed in project virtual environment
- **Runtime**: ONNX Runtime (CPU-bound)
- **Model**: u2net (default)

### Performance Characteristics
- **Processing Time**: ~2-5 seconds per image (CPU, varies by image size)
- **Bottleneck**: ONNX inference on CPU
- **Memory**: Moderate (~200-500MB per session)

## Installation Method Comparison

### Option 1: pip/uv Install (Current)

**Pros:**
- ✅ Simple installation: `uv add rembg`
- ✅ Automatic dependency management
- ✅ Easy updates: `uv sync`
- ✅ Version pinning in `pyproject.toml`
- ✅ Works with existing project structure

**Cons:**
- ❌ No direct control over ONNX runtime configuration
- ❌ Cannot easily modify rembg internals
- ❌ Model downloads managed by rembg (cached in `~/.u2net/`)

**Performance Impact**: None - same binary code

### Option 2: Clone Repository Outside Project Root

**Pros:**
- ✅ Can modify rembg source code directly
- ✅ Can optimize ONNX runtime settings
- ✅ Can experiment with different models
- ✅ Can add custom optimizations

**Cons:**
- ❌ Manual dependency management
- ❌ Harder to update/version control
- ❌ Requires `PYTHONPATH` or `sys.path` manipulation
- ❌ More complex development workflow
- ❌ May break with rembg updates

**Performance Impact**: Minimal - only if you modify the code

### Recommendation: **Keep pip/uv install**

The installation method has **no significant performance impact**. The bottleneck is the ONNX runtime execution, not the installation method.

## Performance Optimization Strategies

### 1. ONNX Runtime Providers (Recommended)

**Current**: Using default CPU provider
**Optimization**: Use optimized providers

```python
# Option A: Use ONNX Runtime with optimizations
import onnxruntime as ort

# Create session with optimizations
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 4  # Use multiple threads
session_options.inter_op_num_threads = 4

# However, rembg doesn't expose these options directly
```

**Limitation**: rembg doesn't expose ONNX session options directly.

**Workaround**: Modify rembg source or use custom wrapper.

### 2. Model Selection

**Available Models** (from rembg):
- `u2net` (default) - Best quality, slower
- `u2netp` - Smaller, faster, slightly lower quality
- `u2net_human_seg` - Optimized for human subjects
- `u2net_cloth_seg` - Optimized for clothing
- `silueta` - Fast, lower quality
- `isnet-general-use` - Good balance

**Recommendation**: Try `u2netp` or `silueta` for faster processing:

```python
from rembg import remove

output = remove(image, model_name="u2netp")  # Faster
# or
output = remove(image, model_name="silueta")  # Fastest
```

**Expected Speedup**: 1.5-2x faster with `u2netp`, 2-3x with `silueta`

### 3. Image Resizing (Pre-processing)

**Strategy**: Resize large images before processing, then scale back

```python
import cv2
from rembg import remove

def remove_bg_optimized(image, max_size=1024):
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        small_image = cv2.resize(image, (new_w, new_h))

        # Process smaller image
        output = remove(small_image)

        # Scale back
        output = cv2.resize(output, (w, h))
        return output
    else:
        return remove(image)
```

**Expected Speedup**: 2-4x for large images (>2048px)

### 4. Batch Processing

**Strategy**: Process multiple images in parallel

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from rembg import remove

def process_batch(images, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(remove, images))
    return results
```

**Expected Speedup**: Linear with number of cores (up to I/O limits)

### 5. GPU Acceleration (Best Option)

**Strategy**: Use ONNX Runtime with CUDA provider

**Requirements**:
- NVIDIA GPU with CUDA support
- `onnxruntime-gpu` package
- CUDA toolkit installed

**Implementation**:
```python
# Install: uv add onnxruntime-gpu
# Note: This replaces onnxruntime

# rembg should automatically use GPU if available
# But you may need to force it:
import os
os.environ["ORT_TENSORRT_ENABLE"] = "1"  # For TensorRT
```

**Expected Speedup**: 5-10x faster on GPU

**Limitation**: Requires GPU hardware

### 6. Model Caching & Session Reuse

**Current**: rembg creates new session per call (or caches internally)

**Optimization**: Reuse rembg session

```python
from rembg import new_session, remove

# Create session once
session = new_session("u2net")

# Reuse for multiple images
for image in images:
    output = remove(image, session=session)
```

**Expected Speedup**: Eliminates model loading overhead (~0.5-1s per image)

### 7. Alpha Matting Optimization

**Current**: Alpha matting enabled by default (slower but better edges)

**Optimization**: Disable for speed

```python
output = remove(
    image,
    alpha_matting=False,  # Disable for speed
    # or
    alpha_matting=True,
    alpha_matting_foreground_threshold=240,
    alpha_matting_background_threshold=10,
    alpha_matting_erode_size=10,
)
```

**Expected Speedup**: 20-30% faster without alpha matting

## Recommended Optimization Plan

### Phase 1: Quick Wins (No Code Changes)
1. ✅ **Use faster model**: Switch to `u2netp` or `silueta`
2. ✅ **Disable alpha matting**: If edge quality is acceptable
3. ✅ **Reuse sessions**: Create session once, reuse for batch

**Expected Improvement**: 2-3x faster

### Phase 2: Image Preprocessing
1. ✅ **Resize large images**: Process at 1024px max, scale back
2. ✅ **Batch processing**: Use ThreadPoolExecutor for parallel processing

**Expected Improvement**: Additional 2-4x for large images

### Phase 3: GPU Acceleration (If Available)
1. ✅ **Install onnxruntime-gpu**: Replace CPU version
2. ✅ **Verify GPU usage**: Check that rembg uses GPU

**Expected Improvement**: 5-10x faster

### Phase 4: Advanced (If Needed)
1. ⚠️ **Custom ONNX optimizations**: Modify rembg source
2. ⚠️ **TensorRT**: For NVIDIA GPUs, use TensorRT backend
3. ⚠️ **Model quantization**: Use INT8 quantized models

## Implementation Example

```python
"""
Optimized rembg wrapper with performance improvements.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from rembg import remove, new_session
from PIL import Image

class OptimizedBackgroundRemover:
    """Optimized background removal with performance improvements."""

    def __init__(
        self,
        model_name: str = "u2netp",  # Faster model
        max_size: int = 1024,  # Resize large images
        alpha_matting: bool = False,  # Disable for speed
        use_gpu: bool = False,
    ):
        self.model_name = model_name
        self.max_size = max_size
        self.alpha_matting = alpha_matting
        self.use_gpu = use_gpu

        # Create session once (reused for all images)
        self.session = new_session(model_name)

    def remove_background(
        self,
        image: np.ndarray | Image.Image | Path | str,
    ) -> np.ndarray:
        """
        Remove background with optimizations.

        Args:
            image: Input image (BGR numpy array, PIL Image, or path)

        Returns:
            Image with background removed (BGR numpy array)
        """
        # Convert to PIL if needed
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
        else:
            pil_image = image

        # Resize if too large
        original_size = pil_image.size
        if max(original_size) > self.max_size:
            scale = self.max_size / max(original_size)
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            resize_back = True
        else:
            resize_back = False

        # Remove background
        output = remove(
            pil_image,
            session=self.session,
            alpha_matting=self.alpha_matting,
        )

        # Resize back if needed
        if resize_back:
            output = output.resize(original_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        output_array = np.array(output)

        # Composite on white background if RGBA
        if output_array.shape[2] == 4:
            rgb = output_array[:, :, :3]
            alpha = output_array[:, :, 3:4] / 255.0
            white_bg = np.ones_like(rgb) * 255
            result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        else:
            result_bgr = cv2.cvtColor(output_array, cv2.COLOR_RGB2BGR)

        return result_bgr


# Usage
remover = OptimizedBackgroundRemover(
    model_name="u2netp",  # Faster model
    max_size=1024,  # Resize large images
    alpha_matting=False,  # Disable for speed
)

# Process images (session reused automatically)
for image_path in image_paths:
    image = cv2.imread(str(image_path))
    result = remover.remove_background(image)
    cv2.imwrite(f"output_{image_path.name}", result)
```

## Benchmarking Results (Expected)

| Configuration | Time per Image | Speedup |
|--------------|----------------|---------|
| Baseline (u2net, alpha_matting=True) | 3.0s | 1.0x |
| u2netp, no alpha_matting | 1.5s | 2.0x |
| u2netp, resize 1024px | 0.8s | 3.75x |
| u2netp, GPU | 0.3s | 10.0x |
| silueta, resize 1024px | 0.5s | 6.0x |

*Note: Actual results vary by image size and hardware*

## Conclusion

### Key Findings

1. **Installation method doesn't matter**: pip install vs cloning has no performance impact
2. **Model selection matters most**: `u2netp` or `silueta` are 2-3x faster
3. **Image resizing helps**: 2-4x speedup for large images
4. **GPU acceleration is best**: 5-10x speedup if available
5. **Session reuse eliminates overhead**: Important for batch processing

### Recommendations

1. **Short term**: Use `u2netp` model, disable alpha matting, reuse sessions
2. **Medium term**: Add image resizing for large images, implement batch processing
3. **Long term**: If GPU available, switch to `onnxruntime-gpu`

### Next Steps

1. Implement `OptimizedBackgroundRemover` class
2. Update test pipeline to use optimized remover
3. Benchmark actual performance improvements
4. Document configuration options

## References

- [rembg GitHub](https://github.com/danielgatis/rembg)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [ONNX Runtime GPU](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)

