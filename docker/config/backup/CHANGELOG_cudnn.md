# Dockerfile cuDNN Installation

**Date**: 2025-11-22
**Change**: Added cuDNN library for GPU acceleration support

## Changes Made

Added `libcudnn9-cuda-12` package to the Dockerfile system dependencies installation.

## Why This Is Needed

- **rembg GPU Acceleration**: The rembg library uses ONNX Runtime with CUDA support for GPU-accelerated background removal
- **ONNX Runtime GPU**: Requires cuDNN (CUDA Deep Neural Network library) to use GPU execution providers
- **Performance**: Enables 5-10x faster background removal processing on GPU

## Package Details

- **Package**: `libcudnn9-cuda-12`
- **Version**: 9.16.0.29 (installed from NVIDIA CUDA repository)
- **CUDA Version**: Compatible with CUDA 12.x
- **Size**: ~441 MB

## Installation

The package is automatically installed when building the Docker image:

```dockerfile
RUN apt-get update && apt-get install -y \
    # ... other packages ...
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*
```

## Verification

After building the image, verify cuDNN is installed:

```bash
docker exec <container> find /usr -name "libcudnn.so*"
# Should show: /usr/lib/x86_64-linux-gnu/libcudnn.so.9
```

## Notes

- The NVIDIA CUDA base image (`nvidia/cuda:12.8.1-devel-ubuntu22.04`) already includes the CUDA repository, so no additional repository setup is needed
- This package is only needed if you plan to use GPU acceleration with rembg
- For CPU-only usage, this package is not required but doesn't hurt to have

