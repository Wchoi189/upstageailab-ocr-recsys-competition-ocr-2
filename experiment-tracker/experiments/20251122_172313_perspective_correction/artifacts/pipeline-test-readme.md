# Test Pipeline: rembg → Perspective Correction

This directory contains scripts for testing the rembg background removal and perspective correction pipeline.

## Quick Start

```bash
# Process 10 sample images
python scripts/test_pipeline_rembg_perspective.py \
    --input-dir data/samples \
    --output-dir outputs/pipeline_test \
    --num-samples 10
```

## Scripts

### `test_pipeline_rembg_perspective.py`

Main test pipeline script that:
1. Removes background using rembg
2. Applies perspective correction
3. Generates sample output images with intermediate results

**Features:**
- Supports multiple image formats (jpg, png, bmp)
- Saves intermediate results (rembg output, final output, comparison)
- Performance metrics (timing for each stage)
- Uses optimized rembg wrapper for better performance

**Usage:**
```bash
python scripts/test_pipeline_rembg_perspective.py \
    --input-dir <input_directory> \
    --output-dir <output_directory> \
    --num-samples <number> \
    [--use-existing-perspective]
```

**Arguments:**
- `--input-dir`: Directory containing input images (default: `data/samples`)
- `--output-dir`: Directory for output images (default: `outputs/pipeline_test`)
- `--num-samples`: Number of images to process (default: 10)
- `--use-existing-perspective`: Use existing PerspectiveCorrector if available (default: True)

### `optimized_rembg.py`

Optimized background removal wrapper that provides:
- Faster model selection (u2netp, silueta)
- Image resizing for large images
- Session reuse for batch processing
- Optional alpha matting

**Usage:**
```python
from scripts.optimized_rembg import OptimizedBackgroundRemover

remover = OptimizedBackgroundRemover(
    model_name="u2netp",  # Faster than u2net
    max_size=1024,  # Resize large images
    alpha_matting=False,  # Disable for speed
)

# Process images (session reused automatically)
result = remover.remove_background(image)
```

## Performance Optimization

See `docs/assessments/rembg_performance_optimization.md` for detailed performance analysis and optimization strategies.

**Key optimizations:**
1. Use `silueta` model (fastest, optimized for speed)
2. Resize to 640px (model training size, optimal performance)
3. Reuse ONNX sessions (eliminates loading overhead)
4. Disable alpha matting (20-30% faster)
5. Use GPU if available (5-10x faster)
6. Use TensorRT for NVIDIA GPUs (additional 1.5-2x speedup)
7. INT8 quantization (if available, 1.5-2x additional speedup)

## Optimized Configuration Testing

### Test Script: `test_optimized_rembg.py`

Tests specific optimization settings:
- **Model**: `silueta` (fastest)
- **Image Size**: 640px (model training size)
- **Alpha Matting**: Disabled
- **GPU/TensorRT**: Enabled if available
- **INT8**: Tested if available

**Usage**:
```bash
python scripts/test_optimized_rembg.py \
    --input-dir data/samples \
    --output-dir outputs/optimized_test \
    --num-samples 10
```

**Features**:
- Automatically tests all available configurations
- Compares CPU vs GPU vs TensorRT performance
- Tests INT8 quantization if available
- Generates performance summary

See `docs/assessments/rembg_optimization_test_plan.md` for detailed test plan.

## Output Structure

```
outputs/pipeline_test/
├── image1_01_rembg.jpg          # After background removal
├── image1_02_final.jpg          # After perspective correction
├── image1_03_comparison.jpg     # Side-by-side comparison
├── image2_01_rembg.jpg
├── image2_02_final.jpg
└── ...
```

## Requirements

- `rembg>=2.0.67`
- `onnxruntime>=1.23.1`
- `opencv-python`
- `numpy`
- `Pillow`

Install with:
```bash
uv sync
```

## References

- [rembg GitHub](https://github.com/danielgatis/rembg)
- [Perspective Correction Reference](https://github.com/sraddhanjali/Automated-Perspective-Correction-for-Scanned-Documents-and-Cards)

