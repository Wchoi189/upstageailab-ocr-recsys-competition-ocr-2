---
ads_version: "1.0"
type: "guide"
experiment_id: "20251122_172313_perspective_correction"
status: "complete"
created: "2025-12-17T17:59:47Z"
updated: "2025-12-17T17:59:47Z"
tags: ['perspective-correction']
commands: []
prerequisites: []
---

# Quick Start: Optimized rembg Testing

## Recommended Configuration

Based on optimization requirements, use these settings:

```python
# Note: optimized_rembg.py is now in experiment_manager
# Update import path as needed for your use case
from optimized_rembg import OptimizedBackgroundRemover

remover = OptimizedBackgroundRemover(
    model_name="silueta",      # Fastest model
    max_size=640,              # Model training size
    alpha_matting=False,       # Disabled for speed
    use_gpu=True,              # Enable GPU if available
    use_tensorrt=True,         # Enable TensorRT if available
    use_int8=False,            # Enable INT8 if models available
)
```

## Quick Test

```bash
# Test optimized configurations
# Note: Script is now in experiment_manager
python experiment_manager/experiments/20251122_172313_perspective_correction/scripts/test_optimized_rembg.py \
    --input-dir data/samples \
    --output-dir outputs/optimized_test \
    --num-samples 10
```

## Check System Capabilities

```bash
python -c "
# Note: Update import path as needed
from optimized_rembg import GPU_AVAILABLE, TENSORRT_AVAILABLE, available_providers
print(f'GPU: {GPU_AVAILABLE}')
print(f'TensorRT: {TENSORRT_AVAILABLE}')
print(f'Providers: {available_providers}')
"
```

## Install GPU Support

```bash
# Install onnxruntime-gpu
uv add onnxruntime-gpu

# Verify
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

## Expected Performance

- **CPU**: ~1.0-2.0s per image
- **GPU (CUDA)**: ~0.2-0.4s per image (5-10x faster)
- **TensorRT**: ~0.1-0.2s per image (10-20x faster)
- **INT8 + TensorRT**: ~0.05-0.1s per image (20-40x faster)

