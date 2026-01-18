This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
data/
  datasets/
    canonical.yaml
    craft.yaml
    recognition.yaml
  runtime/
    performance/
      balanced.yaml
      memory_efficient.yaml
      minimal.yaml
      none.yaml
      README.md
      validation_optimized.yaml
  transforms/
    background_removal.yaml
    base.yaml
    document_geometry.yaml
    image_enhancement.yaml
    recognition.yaml
    with_background_removal.yaml
  default.yaml
domain/
  detection.yaml
  kie.yaml
  layout.yaml
  recognition.yaml
global/
  default.yaml
  paths.yaml
hardware/
  rtx3060.yaml
model/
  presets/
    craft.yaml
    dbnetpp.yaml
    parseq.yaml
train/
  callbacks/
    default.yaml
    early_stopping.yaml
    metadata.yaml
    model_checkpoint.yaml
    model_summary.yaml
    performance_profiler.yaml
    recognition_wandb.yaml
    rich_progress_bar.yaml
    wandb_completion.yaml
    wandb_image_logging.yaml
  logger/
    default.yaml
    wandb.yaml
  optimizer/
    adam.yaml
    adamw.yaml
main.yaml
README.md
```

# Files

## File: data/datasets/canonical.yaml
````yaml
# @package data

defaults:
  - /data/transforms/base@transforms
  - _self_

batch_size: 16

train_num_samples: null
val_num_samples: null
test_num_samples: null

# Dataset Definitions (Flattened)
train_dataset:
  _target_: ocr.data.datasets.ValidatedOCRDataset
  config:
    _target_: ocr.core.validation.DatasetConfig
    image_path: ${global.paths.datasets_root}/images/train
    annotation_path: ${global.paths.datasets_root}/jsons/train.json
    # Explicitly disable caching to prevent default override
    cache_config:
      _target_: ocr.core.validation.CacheConfig
      cache_transformed_tensors: false
      cache_images: false
      cache_maps: false
      log_statistics_every_n: null
  transform: ${data.transforms.train_transform}

val_dataset:
  _target_: ocr.data.datasets.ValidatedOCRDataset
  config:
    _target_: ocr.core.validation.DatasetConfig
    image_path: ${global.paths.datasets_root}/images_val_canonical  # Using canonical (rotation-corrected) data
    annotation_path: ${global.paths.datasets_root}/jsons/val.json
  transform: ${data.transforms.val_transform}

test_dataset:
  _target_: ocr.data.datasets.ValidatedOCRDataset
  config:
    _target_: ocr.core.validation.DatasetConfig
    image_path: ${global.paths.datasets_root}/images_val_canonical  # Using canonical (rotation-corrected) data
    annotation_path: ${global.paths.datasets_root}/jsons/val.json
  transform: ${data.transforms.test_transform}

predict_dataset:
  _target_: ocr.data.datasets.ValidatedOCRDataset
  config:
    _target_: ocr.core.validation.DatasetConfig
    image_path: ${global.paths.datasets_root}/images/test
    annotation_path: null
  transform: ${data.transforms.test_transform}

collate_fn:
  _target_: ocr.data.datasets.DBCollateFN
  shrink_ratio: 0.4
  thresh_min: 0.3
  thresh_max: 0.7
````

## File: data/datasets/craft.yaml
````yaml
# @package _group_

# Standalone CRAFT data configuration

defaults:
  - /data/transforms/base@transforms
  - _self_

batch_size: 16

data:
  train_num_samples: null
  val_num_samples: null
  test_num_samples: null

dataset_base_path: "${global.paths.datasets_root}"

datasets:
  train_dataset:
    _target_: ${dataset_path}.ValidatedOCRDataset
    config:
      _target_: ${dataset_config_path}.DatasetConfig
      image_path: ${dataset_base_path}images/train
      annotation_path: ${dataset_base_path}jsons/train.json
    transform: ${transforms.train_transform}
  val_dataset:
    _target_: ${dataset_path}.ValidatedOCRDataset
    config:
      _target_: ${dataset_config_path}.DatasetConfig
      image_path: ${dataset_base_path}images_val_canonical  # Using canonical (rotation-corrected) data
      annotation_path: ${dataset_base_path}jsons/val.json
    transform: ${transforms.val_transform}
  test_dataset:
    _target_: ${dataset_path}.ValidatedOCRDataset
    config:
      _target_: ${dataset_config_path}.DatasetConfig
      image_path: ${dataset_base_path}images_val_canonical  # Using canonical (rotation-corrected) data
      annotation_path: ${dataset_base_path}jsons/val.json
    transform: ${transforms.test_transform}
  predict_dataset:
    _target_: ${dataset_path}.ValidatedOCRDataset
    config:
      _target_: ${dataset_config_path}.DatasetConfig
      image_path: ${dataset_base_path}images/test
      annotation_path: null
    transform: ${transforms.test_transform}

transforms:
  train_transform:
    _target_: ${dataset_path}.DBTransforms
    transforms:
      - _target_: albumentations.LongestMaxSize
        max_size: 768
        p: 1.0
      - _target_: albumentations.PadIfNeeded
        min_width: 768
        min_height: 768
        border_mode: 0
        p: 1.0
      - _target_: albumentations.RandomRotate90
        p: 0.5
      - _target_: albumentations.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    keypoint_params:
      _target_: albumentations.KeypointParams
      format: 'xy'
      remove_invisible: True
  val_transform:
    _target_: ${dataset_path}.DBTransforms
    transforms:
      - _target_: albumentations.LongestMaxSize
        max_size: 768
        p: 1.0
      - _target_: albumentations.PadIfNeeded
        min_width: 768
        min_height: 768
        border_mode: 0
        p: 1.0
      - _target_: albumentations.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    keypoint_params:
      _target_: albumentations.KeypointParams
      format: 'xy'
      remove_invisible: True
  test_transform:
    _target_: ${dataset_path}.DBTransforms
    transforms:
      - _target_: albumentations.LongestMaxSize
        max_size: 768
        p: 1.0
      - _target_: albumentations.PadIfNeeded
        min_width: 768
        min_height: 768
        border_mode: 0
        p: 1.0
      - _target_: albumentations.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    keypoint_params:
      _target_: albumentations.KeypointParams
      format: 'xy'
      remove_invisible: True
  predict_transform:
    _target_: ${dataset_path}.DBTransforms
    transforms:
      - _target_: albumentations.LongestMaxSize
        max_size: 768
        p: 1.0
      - _target_: albumentations.PadIfNeeded
        min_width: 768
        min_height: 768
        border_mode: 0
        p: 1.0
      - _target_: albumentations.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    keypoint_params: null

collate_fn:
  _target_: ${dataset_path}.CraftCollateFN
  region_blur_scale: 0.35
  affinity_kernel_ratio: 0.25
  affinity_blur_scale: 0.15
  min_text_area: 9
````

## File: data/datasets/recognition.yaml
````yaml
# @package data
# Recognition Dataset Configuration
# Used for text recognition models (PARSeq)

defaults:
  - /data/transforms/recognition@transforms

# Data Globals
batch_size: 128
num_workers: 4

# Tokenizer
tokenizer:
  _target_: ocr.features.recognition.data.tokenizer.KoreanOCRTokenizer
  charset_path: ${global.paths.root_dir}/ocr/data/charset.json
  max_len: 25

# Dataset Template
_dataset_template: &dataset_template
  _target_: ocr.features.recognition.data.lmdb_dataset.LMDBRecognitionDataset
  lmdb_path: ${global.paths.datasets.train}/recognition/aihub_lmdb_validation
  tokenizer: ${data.tokenizer}
  max_len: ${data.tokenizer.max_len}
  transform: ${data.transforms}
  config: null  # Suppress inherited 'config' key from base.yaml merge

# Explicit dataset definitions for factory (at root/datasets)
datasets:
  train_dataset:
    <<: *dataset_template

  val_dataset:
    <<: *dataset_template

  test_dataset:
    <<: *dataset_template

  predict_dataset:
    <<: *dataset_template

# Collate (at root)
collate_fn:
  _target_: ocr.data.datasets.recognition_collate_fn.recognition_collate_fn
  _partial_: true
````

## File: data/runtime/performance/balanced.yaml
````yaml
# @package _group_.datasets.val_dataset.config
# Performance Preset: Balanced Performance
#
# Use Case: Standard training workflows with moderate memory
# Memory: ~2.4 GB (+400 MB vs minimal)
# Speed: ~1.12x (10-12% faster)
# Safe for Training: ✅ YES
#
# This preset enables image caching for moderate speedup without
# risk of data leakage. Good default for production training.

# Image loading - no preloading to save startup time
preload_images: false
preload_maps: false
prenormalize_images: false

# Map loading - disabled for training safety
load_maps: false

# Cache configuration - image cache only
cache_config:
  cache_images: true  # Cache decoded images
  cache_maps: false
  cache_transformed_tensors: false  # NEVER enable for training!
  log_statistics_every_n: 100  # Monitor cache health

# Image loading configuration
image_loading_config:
  use_turbojpeg: true
  turbojpeg_fallback: true
````

## File: data/runtime/performance/memory_efficient.yaml
````yaml
# @package _group_.datasets.val_dataset.config
# Performance Preset: Memory Efficient
#
# Use Case: Low RAM systems (< 4 GB available), large datasets
# Memory: ~2.0 GB (same as minimal, optimized for low overhead)
# Speed: 1.0x (baseline, optimized for memory not speed)
# Safe for Training: ✅ YES
#
# This preset minimizes memory usage at the cost of speed.
# All caching disabled, no preloading, conservative settings.

# Image loading - no preloading
preload_images: false
preload_maps: false
prenormalize_images: false

# Map loading - disabled to save memory
load_maps: false

# Cache configuration - all disabled to save memory
cache_config:
  cache_images: false  # No image caching
  cache_maps: false
  cache_transformed_tensors: false
  log_statistics_every_n: null  # No logging overhead

# Image loading configuration
image_loading_config:
  use_turbojpeg: true  # Fast loading still helps
  turbojpeg_fallback: true
````

## File: data/runtime/performance/minimal.yaml
````yaml
# @package _group_.datasets.val_dataset.config
# Performance Preset: Minimal Overhead (Default)
#
# Use Case: Development, debugging, memory-constrained environments
# Memory: ~2.0 GB (baseline)
# Speed: 1.0x (baseline)
# Safe for Training: ✅ YES
#
# This is the DEFAULT preset with minimal overhead and maximum compatibility.
# All optimizations disabled for predictable behavior and easy debugging.

# Image loading - no preloading
preload_images: false
preload_maps: false
prenormalize_images: false

# Map loading - disabled
load_maps: false

# Cache configuration - all disabled
cache_config:
  cache_images: false
  cache_maps: false
  cache_transformed_tensors: false
  log_statistics_every_n: null  # No cache logging

# Image loading configuration
image_loading_config:
  use_turbojpeg: true  # Still use fast JPEG loading
  turbojpeg_fallback: true
````

## File: data/runtime/performance/none.yaml
````yaml
# @package _group_

defaults:
  - _self_

# Default runtime settings (no specific optimizations)
caching: false
prefetch: false
````

## File: data/runtime/performance/README.md
````markdown
# Performance Presets for Dataset Configuration

This directory contains preset configurations for different performance optimization levels. Use these to easily toggle between different combinations of caching, preloading, and other performance features.

## Available Presets

### 1. `none.yaml` - No Optimizations (Explicit Default)
**Use Case**: Explicitly disable all performance features
- Identical to `minimal` but provided for clarity
- Use when you want to explicitly state "no optimizations"
- Useful for documentation and debugging

### 2. `minimal.yaml` - Minimal Overhead (Default)
**Use Case**: Development, debugging, memory-constrained environments
- No caching
- No preloading
- Minimal memory footprint
- Baseline performance

### 3. `balanced.yaml` - Balanced Performance
**Use Case**: Standard training, moderate memory available
- Image caching enabled
- No tensor caching (safe for training)
- Moderate memory usage
- ~10-15% speedup

### 4. `validation_optimized.yaml` - Validation Speed
**Use Case**: Validation/test datasets only (NOT training!)
- Full caching (images + tensors + maps)
- Image preloading
- Map preloading
- Maximum speedup (~2.5-3x after warm-up)
- High memory usage

### 5. `memory_efficient.yaml` - Memory Constrained
**Use Case**: Low RAM systems, large datasets
- Minimal caching
- No preloading
- Optimized for memory
- Slight performance trade-off

## Usage

### Override in Command Line

```bash
# Use none preset (explicit no optimizations)
uv run python runners/train.py data/performance_preset=none

# Use minimal preset (default - same as 'none')
uv run python runners/train.py data/performance_preset=minimal

# Use balanced preset
uv run python runners/train.py data/performance_preset=balanced

# Use validation optimized (for validation dataset only!)
uv run python runners/train.py data/performance_preset=validation_optimized

# Use memory efficient
uv run python runners/train.py data/performance_preset=memory_efficient
```

### Override in Config File

```yaml
# configs/train.yaml
defaults:
  - data/performance_preset: balanced
```

### Per-Dataset Override

```yaml
# configs/data/base.yaml
datasets:
  train_dataset:
    config:
      defaults:
        - /data/performance_preset: minimal  # Safe for training

  val_dataset:
    config:
      defaults:
        - /data/performance_preset: validation_optimized  # Fast validation
```

## Preset Comparison Table

| Feature | None | Minimal | Balanced | Validation Optimized | Memory Efficient |
|---------|------|---------|----------|---------------------|------------------|
| **Memory Usage** | ~2.0 GB | ~2.0 GB | ~2.4 GB | ~3.5 GB | ~2.0 GB |
| **Speedup** | 1.0x | 1.0x | ~1.12x | ~2.5-3x (after warm-up) | 1.0x |
| **Epoch 0 Time** | Baseline | Baseline | +5% | +20% (cache build) | Baseline |
| **Epoch 1+ Time** | Baseline | Baseline | -10% | -60% to -70% | Baseline |
| **Cache Images** | ❌ | ❌ | ✅ | ✅ | ❌ |
| **Cache Tensors** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Cache Maps** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Preload Images** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Preload Maps** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Load Maps** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Safe for Training** | ✅ | ✅ | ✅ | ❌ | ✅ |

## Performance Guidelines

### When to Use Each Preset

#### Minimal (Default)
- ✅ Development and debugging
- ✅ Memory-constrained systems
- ✅ Training datasets (always safe)
- ✅ Investigating data issues
- ❌ Production training (too slow)

#### Balanced
- ✅ Standard training workflows
- ✅ Moderate memory available (4-6 GB)
- ✅ Good performance/memory trade-off
- ✅ Training datasets (safe)
- ❌ Need maximum speed

#### Validation Optimized
- ✅ Validation/test datasets ONLY
- ✅ Maximum speed required
- ✅ Sufficient memory available (6+ GB)
- ❌ **NEVER use for training datasets** (data leakage risk!)
- ❌ Memory-constrained systems

#### Memory Efficient
- ✅ Low RAM systems (< 4 GB available)
- ✅ Very large datasets
- ✅ Debugging memory issues
- ❌ Need fast training

## Safety Warnings

### ⚠️ CRITICAL: Tensor Caching on Training Data

**DO NOT enable `cache_transformed_tensors` on training datasets!**

This causes data leakage because:
1. Augmentations (random crops, flips) are cached
2. Same augmented samples repeated every epoch
3. Model overfits to cached augmentations
4. Validation metrics look good but model doesn't generalize

**Safe Usage**:
- ✅ Validation datasets (no augmentation)
- ✅ Test datasets (no augmentation)
- ❌ Training datasets (augmentation present)

### ⚠️ Memory Usage

- **Validation Optimized**: Uses ~1.5 GB extra RAM
- **Monitor Usage**: `uv run python scripts/cache_manager.py status`
- **Clear Cache**: `uv run python scripts/cache_manager.py clear --all`

## Customization

Create your own preset by copying an existing one:

```bash
cp configs/data/performance_preset/balanced.yaml \
   configs/data/performance_preset/custom.yaml
```

Then edit `custom.yaml` to your needs:

```yaml
# configs/data/performance_preset/custom.yaml
# @package _global_.datasets.val_dataset.config

# My custom preset
cache_config:
  cache_images: true
  cache_tensors: false  # Safe setting
  cache_maps: true
  log_statistics_every_n: 50  # More frequent logging

preload_images: false  # Save startup time
load_maps: true        # Enable evaluation
```

## Troubleshooting

### High Memory Usage

If you experience OOM (Out of Memory) errors:

1. Switch to `memory_efficient` preset:
   ```bash
   uv run python runners/train.py data/performance_preset=memory_efficient
   ```

2. Clear cache:
   ```bash
   uv run python scripts/cache_manager.py clear --all
   ```

3. Monitor memory:
   ```bash
   watch -n 1 nvidia-smi  # GPU memory
   htop  # CPU memory
   ```

### Slow First Epoch

This is expected with `validation_optimized` preset:
- Epoch 0: Builds cache (20% slower)
- Epoch 1+: Uses cache (2.5-3x faster)

**Trade-off is worth it for multi-epoch training!**

### Cache Not Working

Check cache health:
```bash
uv run python scripts/cache_manager.py health
```

Look for:
- Cache version mismatches
- Stale files (> 7 days old)
- Low hit rates (< 50%)

### Stale Cache

Clear and rebuild:
```bash
uv run python scripts/cache_manager.py clear --all
uv run python runners/train.py  # Rebuilds cache
```

## Related Documentation

- [Cache Management Guide](../../../docs/performance/CACHE_MANAGEMENT_GUIDE.md)
- [Performance Benchmark Commands](../../../docs/performance/BENCHMARK_COMMANDS.md)
- [Dataset Configuration Reference](../base.yaml)

## Examples

### Example 1: Development Workflow

```bash
# Use minimal for quick iteration
uv run python runners/train.py \
  data/performance_preset=minimal \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=10
```

### Example 2: Production Training

```yaml
# configs/train_production.yaml
defaults:
  - override /data/performance_preset: balanced  # Training dataset
  - override /data/performance_preset@datasets.val_dataset: validation_optimized  # Validation dataset
```

### Example 3: Memory-Constrained System

```bash
# Use memory efficient preset
uv run python runners/train.py \
  data/performance_preset=memory_efficient \
  batch_size=8  # Reduce batch size too
```

## Version History

- **v1.0** (2025-10-14): Initial presets (minimal, balanced, validation_optimized, memory_efficient)
- Future: Add GPU-specific presets, dynamic preset selection

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section
- Review [Cache Management Guide](../../../docs/performance/CACHE_MANAGEMENT_GUIDE.md)
- Run cache health check: `uv run python scripts/cache_manager.py health`
````

## File: data/runtime/performance/validation_optimized.yaml
````yaml
# @package _group_.datasets.val_dataset.config
# Performance Preset: Validation Optimized
#
# Use Case: Validation/test datasets ONLY (NOT training!)
# Memory: ~3.5 GB (+1.5 GB vs minimal)
# Speed: ~2.5-3x after cache warm-up (Epoch 0: +20%, Epoch 1+: -60% to -70%)
# Safe for Training: ❌ NO - Data leakage risk!
#
# ⚠️ CRITICAL WARNING:
# ONLY use this preset for validation/test datasets!
# NEVER use for training datasets - causes data leakage!
#
# This preset maximizes speed by caching everything, including
# transformed tensors. Only safe when NO augmentation is present.

# Image loading - preload everything
preload_images: true  # Load all images to RAM at startup
preload_maps: true    # Load all maps to RAM at startup
prenormalize_images: false

# Map loading - enabled for evaluation
load_maps: true

# Cache configuration - FULL caching enabled
cache_config:
  cache_images: true              # Cache decoded images
  cache_maps: true                # Cache probability/threshold maps
  cache_transformed_tensors: true # ⚠️ ONLY safe for validation!
  log_statistics_every_n: 100     # Monitor cache performance

# Image loading configuration
image_loading_config:
  use_turbojpeg: true
  turbojpeg_fallback: true


# What Makes It Unsafe for Training
# The preset enables cache_transformed_tensors: true, which caches the fully transformed/preprocessed tensors to disk/RAM. This is problematic for training datasets because:

# Data Augmentation Bypass: During training, you want random augmentations (rotations, crops, noise, etc.) applied fresh each epoch to prevent overfitting
# Data Leakage: If transformations are cached, the model sees identical samples every epoch instead of varied augmentations
# Reduced Generalization: The model memorizes specific transformed versions rather than learning robust features
# Safe Usage in Full Training Runs
# You CAN use this preset in complete training runs (train + validate + test) because:

# The preset is scoped to# @package _group_
# It only affects validation/test datasets, not training datasets
# Training data remains unaffected and gets fresh augmentations each epoch
# Validation/test data benefits from cached deterministic transformations (faster, consistent evaluation)
# When You'd Want to Avoid It
# If you have data augmentation on validation sets (uncommon, but possible)
# If you need to evaluate on completely fresh transformations each time
# For debugging transformation pipelines
# The preset is designed specifically for validation acceleration while keeping training data augmentation intact. Your training runs with this preset are perfectly safe - it just makes validation much faster (~2.5-3x speedup after cache warmup).
````

## File: data/transforms/background_removal.yaml
````yaml
# @package _group_
# Background removal transform configurations
# These can be included in your transform pipelines

# Basic background removal (fast, good quality)
background_removal_basic:
  _target_: ocr.data.datasets.preprocessing.external.BackgroundRemoval
  model: "u2net"
  alpha_matting: false
  p: 1.0

# High-quality background removal with alpha matting
background_removal_high_quality:
  _target_: ocr.data.datasets.preprocessing.external.BackgroundRemoval
  model: "u2net"
  alpha_matting: true
  alpha_matting_foreground_threshold: 240
  alpha_matting_background_threshold: 10
  alpha_matting_erode_size: 10
  p: 1.0

# Lightweight model for faster processing
background_removal_fast:
  _target_: ocr.data.datasets.preprocessing.external.BackgroundRemoval
  model: "u2netp"
  alpha_matting: false
  p: 1.0

# Only generate mask (for debugging or custom compositing)
background_removal_mask_only:
  _target_: ocr.data.datasets.preprocessing.external.BackgroundRemoval
  model: "u2net"
  only_mask: true
  p: 1.0

# Convenience function for easy configuration
background_removal_default:
  _target_: ocr.data.datasets.preprocessing.external.create_background_removal_transform
  model: "u2net"
  alpha_matting: true
  p: 1.0
````

## File: data/transforms/base.yaml
````yaml
# @package _group_
# v5.0 Standard | Purpose: Atomic DBNet Transforms (Flattened)
# ============================================================================

# Rule: No top-level 'transforms:' wrapper key
# Rule: Hardcoded or absolute interpolation for constants
train_transform:
  # Rule: _target_ at column 0 and top of block
  _target_: ocr.data.datasets.DBTransforms
  transforms:
    - _target_: albumentations.LongestMaxSize
      max_size: 640
      interpolation: ${global.default_interpolation} # Absolute Global Anchor
      p: 1.0
    - _target_: albumentations.PadIfNeeded
      min_width: 640
      min_height: 640
      border_mode: 0
      position: "top_left"
      p: 1.0
    - _target_: albumentations.HorizontalFlip
      p: 0.5
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  keypoint_params:
    _target_: albumentations.KeypointParams
    format: 'xy'
    remove_invisible: True
val_transform:
  _target_: ocr.data.datasets.DBTransforms
  transforms:
# BugRef: BUG-2025-001 — enforce top-left padding to simplify inverse mapping
    # Report: docs/bug_reports/BUG-2025-001_inference_padding_scaling_mismatch.md
    # Date: 2025-10-20
    - _target_: albumentations.LongestMaxSize
      max_size: 640
      interpolation: 1
      p: 1.0
    - _target_: albumentations.PadIfNeeded
      min_width: 640
      min_height: 640
      border_mode: 0
      position: "top_left"
      p: 1.0
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  keypoint_params:
    _target_: albumentations.KeypointParams
    format: 'xy'
    remove_invisible: True
test_transform:
  _target_: ocr.data.datasets.DBTransforms
  transforms:
# BugRef: BUG-2025-001 — enforce top-left padding to simplify inverse mapping
    # Report: docs/bug_reports/BUG-2025-001_inference_padding_scaling_mismatch.md
    # Date: 2025-10-20
    - _target_: albumentations.LongestMaxSize
      max_size: 640
      interpolation: 1
      p: 1.0
    - _target_: albumentations.PadIfNeeded
      min_width: 640
      min_height: 640
      border_mode: 0
      position: "top_left"
      p: 1.0
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  keypoint_params:
    _target_: albumentations.KeypointParams
    format: 'xy'
    remove_invisible: True
predict_transform:
  _target_: ocr.data.datasets.DBTransforms
  transforms:
# BugRef: BUG-2025-001 — enforce top-left padding to simplify inverse mapping
    # Report: docs/bug_reports/BUG-2025-001_inference_padding_scaling_mismatch.md
    # Date: 2025-10-20
    - _target_: albumentations.LongestMaxSize
      max_size: 640
      interpolation: 1  # cv2.INTER_LINEAR for faster processing
      p: 1.0
    - _target_: albumentations.PadIfNeeded
      min_width: 640
      min_height: 640
      border_mode: 0
      position: "top_left"
      p: 1.0
    - _target_: albumentations.Normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  keypoint_params: null
````

## File: data/transforms/document_geometry.yaml
````yaml
# @package _group_
# v5.0 Standard | Purpose: Document Detection & Rectification
# ============================================================================

_target_: ocr.core.transforms.DocumentGeometryTransform

# --- DocumentDetector ---
enable_detection: true
min_area_ratio: 0.18
use_adaptive: true
use_fallback_box: true

# --- PerspectiveCorrector ---
enable_rectification: true
use_doctr_geometry: false
doctr_assume_horizontal: false

# --- OrientationCorrector ---
enable_orientation_correction: false
angle_threshold: 2.0
expand_canvas: true
````

## File: data/transforms/image_enhancement.yaml
````yaml
# @package _group_
# v5.0 Standard | Purpose: Visual Quality & Resizing
# ============================================================================

_target_: ocr.core.transforms.ImageEnhancementTransform

# --- ImageEnhancer ---
enable_enhancement: true
method: "conservative"  # Options: [conservative, office_lens]

# --- FinalResizer ---
target_size: [640, 640]
enable_final_resize: true

# --- PaddingCleanup ---
enable_padding_cleanup: false
````

## File: data/transforms/recognition.yaml
````yaml
# @package _group_
# Recognition Transforms
# Simple transforms for text recognition (resize + normalize)

_target_: torchvision.transforms.Compose
transforms:
  - _target_: torchvision.transforms.Resize
    size: [224, 224]  # H, W for PARSeq (Temporarily increased for Debugging)
  - _target_: torchvision.transforms.ToTensor
  - _target_: torchvision.transforms.Normalize
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
````

## File: data/transforms/with_background_removal.yaml
````yaml
# @package _group_

# Example configuration showing how to integrate background removal
# into your OCR preprocessing pipeline

# Import base transforms
defaults:
  - base

# Override transforms to include background removal
transforms:
  # Training transform with background removal
  train_transform:
    _target_: ${dataset_path}.DBTransforms
    transforms:
      # Background removal should be applied early in the pipeline
      - _target_: ocr.data.datasets.preprocessing.external.BackgroundRemoval
        model: "u2net"
        alpha_matting: true
        p: 1.0  # Apply to all training images
      - _target_: albumentations.LongestMaxSize
        max_size: 640
        interpolation: ${default_interpolation}
        p: 1.0
      - _target_: albumentations.PadIfNeeded
        min_width: 640
        min_height: 640
        border_mode: 0
        p: 1.0
      - _target_: albumentations.HorizontalFlip
        p: 0.5
      - _target_: albumentations.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    keypoint_params:
      _target_: albumentations.KeypointParams
      format: 'xy'
      remove_invisible: True

  # Validation transform with background removal
  val_transform:
    _target_: ${dataset_path}.DBTransforms
    transforms:
      - _target_: ocr.data.datasets.preprocessing.external.BackgroundRemoval
        model: "u2net"
        alpha_matting: true
        p: 1.0
      - _target_: albumentations.LongestMaxSize
        max_size: 640
        interpolation: ${default_interpolation}
        p: 1.0
      - _target_: albumentations.PadIfNeeded
        min_width: 640
        min_height: 640
        border_mode: 0
        p: 1.0
      - _target_: albumentations.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    keypoint_params:
      _target_: albumentations.KeypointParams
      format: 'xy'
      remove_invisible: True

  # Test transform with background removal
  test_transform:
    _target_: ${dataset_path}.DBTransforms
    transforms:
      - _target_: ocr.data.datasets.preprocessing.external.BackgroundRemoval
        model: "u2net"
        alpha_matting: true
        p: 1.0
      - _target_: albumentations.LongestMaxSize
        max_size: 640
        interpolation: ${default_interpolation}
        p: 1.0
      - _target_: albumentations.PadIfNeeded
        min_width: 640
        min_height: 640
        border_mode: 0
        p: 1.0
      - _target_: albumentations.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    keypoint_params:
      _target_: albumentations.KeypointParams
      format: 'xy'
      remove_invisible: True

  # Prediction transform with background removal
  predict_transform:
    _target_: ${dataset_path}.DBTransforms
    transforms:
      - _target_: ocr.data.datasets.preprocessing.external.BackgroundRemoval
        model: "u2net"
        alpha_matting: true
        p: 1.0
      - _target_: albumentations.LongestMaxSize
        max_size: 640
        interpolation: 1
        p: 1.0
      - _target_: albumentations.PadIfNeeded
        min_width: 640
        min_height: 640
        border_mode: 0
        p: 1.0
      - _target_: albumentations.Normalize
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
    keypoint_params: null
````

## File: data/default.yaml
````yaml
# @package _group_

defaults:
  - /data/transforms/base@transforms
  - _self_

batch_size: 4

train_num_samples: null
val_num_samples: null
test_num_samples: null

# Dataset Definitions (Flattened)
train_dataset:
  _target_: ocr.data.datasets.ValidatedOCRDataset
  config:
    _target_: ocr.core.validation.DatasetConfig
    image_path: ${global.paths.datasets_root}/images/train
    annotation_path: ${global.paths.datasets_root}/jsons/train.json
    enable_background_normalization: false
    cache_config:
      _target_: ocr.core.validation.CacheConfig
      cache_transformed_tensors: false
      cache_images: false
      cache_maps: false
      log_statistics_every_n: null
  transform: ${transforms.train_transform}

val_dataset:
  _target_: ocr.data.datasets.ValidatedOCRDataset
  config:
    _target_: ocr.core.validation.DatasetConfig
    image_path: ${global.paths.datasets_root}/images_val_canonical
    annotation_path: ${global.paths.datasets_root}/jsons/val.json
    enable_background_normalization: false
    preload_images: false
    load_maps: false
    cache_config:
      _target_: ocr.core.validation.CacheConfig
      cache_transformed_tensors: false
      cache_images: false
      cache_maps: false
      log_statistics_every_n: null
  transform: ${transforms.val_transform}

test_dataset:
  _target_: ocr.data.datasets.ValidatedOCRDataset
  config:
    _target_: ocr.core.validation.DatasetConfig
    image_path: ${global.paths.datasets_root}/images_val_canonical
    annotation_path: ${global.paths.datasets_root}/jsons/val.json
    enable_background_normalization: false
  transform: ${transforms.test_transform}

predict_dataset:
  _target_: ocr.data.datasets.ValidatedOCRDataset
  config:
    _target_: ocr.core.validation.DatasetConfig
    image_path: ${global.paths.datasets_root}/images/test
    annotation_path: null
    enable_background_normalization: false
  transform: ${transforms.test_transform}

collate_fn:
  _target_: ocr.data.datasets.DBCollateFN
  shrink_ratio: 0.4
  thresh_min: 0.3
  thresh_max: 0.7
````

## File: domain/detection.yaml
````yaml
# @package _group_

# Text Detection Domain Configuration
# Use with: python runners/train.py +domain=detection

defaults:
  - /global/default
  - /model/presets: dbnetpp
  - /data/datasets/canonical
  - /data/transforms/document_geometry@document_geometry
  - /data/transforms/image_enhancement@image_enhancement
  - _self_

# Rule: _self_ at bottom ensures domain-specific overrides win
# Domain-specific logic
task: detection
batch_size: 16
skip_test: true

model:
  type: detection
  task_type: detection

data:
  task_type: detection

# ============================================================================
# Domain Isolation - Nullify Other Domains
# ============================================================================
recognition: null
max_label_length: null
charset: null
decode_mode: null
beam_width: null

kie: null
max_entities: null
relation_types: null
entity_categories: null
linking_mode: null
````

## File: domain/kie.yaml
````yaml
# @package _group_

# Key Information Extraction (KIE) Domain Configuration
# Use with: python runners/train.py +domain=kie

defaults:
  - /extraction: default
  - _self_

task: kie

model:
  type: kie
  use_ocr_features: true
  use_layout_features: true
  # LayoutLMv3 configuration
  pretrained_model_name_or_path: "microsoft/layoutlmv3-base"
  num_labels: 32  # 31 entities + 1 (O) - simple format, not BIO
  max_length: 512

data:
  task_type: kie
  include_relations: true
  # Default paths - can be overridden via CLI
  train_path: "data/processed/aligned/baseline_kie_dp_train.parquet"
  val_path: "data/processed/aligned/baseline_kie_dp_val.parquet"
  image_dir: "."
  # Label list: O + 31 entity types
  label_list: [
    "O",
    "approval_code", "branch_name", "card_payment_price", "cash_payment_price",
    "cc_code", "cc_expiry", "cc_number", "change", "charged_price",
    "coupon_payment_price", "group_0", "installment", "payment_price",
    "product_name", "store_address", "store_name", "store_phone_number",
    "store_registration_number", "subtotal_price", "tax_price", "tip_price",
    "transaction_date", "unit_product_code", "unit_product_discounted_price",
    "unit_product_discounted_reason", "unit_product_order", "unit_product_price",
    "unit_product_quantity", "unit_product_total_price_after_discount",
    "unit_product_total_price_before_discount", "voucher_payment_price"
  ]

train:
  batch_size: 12
  num_workers: 8
  max_epochs: 15
  warmup_steps: 100
````

## File: domain/layout.yaml
````yaml
# @package _group_

# Layout Analysis Domain Configuration
# Use with: python runners/train.py +domain=layout

defaults:
  - /layout: default
  - _self_

task: layout_analysis

model:
  type: layout

data:
  task_type: layout

# Layout detection configuration
layout:
  enabled: true
  grouping:
    backend: "rule_based"
    y_overlap_threshold: 0.5
    x_gap_ratio: 1.5
    merge_close_lines: false
    line_merge_threshold: 0.3
  hierarchy:
    detect_tables: false
    detect_headers: false
    detect_lists: false
````

## File: domain/recognition.yaml
````yaml
# @package _group_
# v5.0 | Purpose: Master Controller for Text Recognition (PARSeq)
# Validated: 2026-01-18 | Deps: model/presets/parseq, data/recognition
# ============================================================================
# DOMAIN CONTROLLER TEMPLATE - Recognition Example
# ============================================================================
# Purpose: Serve as the master logic switch for a specific domain
# Location: configs/domain/recognition.yaml
#
# This file demonstrates the "Domain Controller" pattern that enforces
# strict domain isolation and prevents configuration leakage.
# ============================================================================

defaults:
  # Component selections for this domain
  - /global/default
  - /global/paths
  - /model/presets/parseq
  - /data/datasets: recognition
  - _self_

  # Optional: training configuration

# ============================================================================
# CRITICAL: Domain Isolation - Nullify Other Domains
# ============================================================================
# These explicit nullifications prevent "ghost variables" that cause:
# - CUDA segfaults
# - Unexpected behavior
# - AI agent confusion
#
# Rule: If a key belongs to another domain, it MUST be set to null here.
# ============================================================================

# Nullify Detection Domain
detection: null
max_polygons: null
shrink_ratio: null
thresh_min: null
thresh_max: null
box_thresh: null
unclip_ratio: null

# Nullify KIE Domain
kie: null
max_entities: null
relation_types: null
entity_categories: null
linking_mode: null

# ============================================================================
# Recognition-Specific Configuration
# ============================================================================

recognition:
  # Character recognition settings
  max_label_length: 25
  charset: korean
  case_sensitive: false

  # Decoding strategy
  decode_mode: greedy  # Options: greedy, beam_search
  beam_width: 5        # Only used if decode_mode=beam_search

  # Special tokens
  use_eos_token: true
  use_pad_token: true

# ============================================================================
# Domain Metadata (Optional but Recommended)
# ============================================================================

domain_info:
  name: recognition
  description: "Korean text recognition using PARSeq architecture"
  version: "2.0"

  # Validation rules for runtime checks
  required_keys:
    - recognition.max_label_length
    - recognition.charset

  forbidden_keys:
    - max_polygons
    - max_entities

# ============================================================================
# Adaptation Guide for Other Domains
# ============================================================================
````

## File: global/default.yaml
````yaml
# @package _global_

defaults:
  - paths
  - _self_

# Global Constants
global:
  default_interpolation: 1

# Global Trainer Defaults
trainer:
  max_steps: -1
  max_epochs: 3
  num_sanity_val_steps: 1
  log_every_n_steps: 20
  val_check_interval: null
  check_val_every_n_epoch: 1
  deterministic: false
  accumulate_grad_batches: 1
  precision: "32-true"
  benchmark: false
  gradient_clip_val: 1.0
  accelerator: auto
  devices: 1
  strategy: auto
  enable_checkpointing: false
  limit_train_batches: null
  limit_val_batches: null
  limit_test_batches: null

# Global Dataloader Defaults (Fallback)
dataloaders:
  train_dataloader:
    num_workers: 4
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 2
  val_dataloader:
    num_workers: 4
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 2
  test_dataloader:
    num_workers: 2
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 1
  predict_dataloader:
    num_workers: 2
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 1
````

## File: global/paths.yaml
````yaml
# @package _global_
# ============================================================================
# GLOBAL PATHS CONFIGURATION - The Single Source of Truth
# ============================================================================
# Purpose: Centralize ALL path interpolations to eliminate "hidden path" problems
# Usage:   This file is automatically included via defaults in config.yaml
#
# CRITICAL: This is the ONLY file that should define path variables.
#           All other configs MUST reference these paths, never redefine them.
# ============================================================================

global:
  paths:
    # ========================================================================
    # Base Directories
    # ========================================================================
    root_dir: .
    data_dir: ${global.paths.root_dir}/data
    output_dir: ${global.paths.root_dir}/outputs

    # Dataset Root (Added for V5 Refactor)
    datasets_root: ${global.paths.data_dir}/datasets

    # ========================================================================
    # Domain-Specific Output Paths
    # ========================================================================
    # Each domain gets its own isolated output directory
    artifacts:
      detection: ${global.paths.output_dir}/detection
      recognition: ${global.paths.output_dir}/recognition
      kie: ${global.paths.output_dir}/kie

    # ========================================================================
    # Training Artifacts
    # ========================================================================
    checkpoints: ${global.paths.output_dir}/checkpoints
    logs: ${global.paths.output_dir}/logs
    wandb: ${global.paths.output_dir}/wandb

    # ========================================================================
    # Data Subdirectories
    # ========================================================================
    datasets:
      train: ${global.paths.data_dir}/train
      val: ${global.paths.data_dir}/val
      test: ${global.paths.data_dir}/test

    # ========================================================================
    # Cache and Temporary Storage
    # ========================================================================
    cache: ${global.paths.output_dir}/cache
    temp: ${global.paths.output_dir}/temp

# ============================================================================
# Hydra Runtime Configuration
# ============================================================================
# Override Hydra's default output directories to use our centralized paths
hydra:
  run:
    # Organize runs by date and time
    dir: ${global.paths.output_dir}/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}

  sweep:
    # Organize sweeps by date and time
    dir: ${global.paths.output_dir}/sweeps/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}

# ============================================================================
# Usage Examples
# ============================================================================
#
# In other config files, reference paths like this:
#
# # configs/data/recognition/lmdb.yaml
# data:
#   train_path: ${global.paths.datasets.train}/recognition_lmdb
#   val_path: ${global.paths.datasets.val}/recognition_lmdb
#
# # configs/model/recognition/parseq.yaml
# model:
#   checkpoint_dir: ${global.paths.checkpoints}/parseq
#
# NEVER do this:
# ❌ data_dir: ./data  # Hardcoded path
# ❌ output_dir: /tmp/outputs  # Absolute path
# ❌ train_path: ${data_dir}/train  # Undefined variable
#
# ============================================================================
````

## File: hardware/rtx3060.yaml
````yaml
# @package _global_

# Hardware capabilities
hardware:
  gpu: "rtx3060"
  vram: "12gb"
  cpu: "i5_16core"

# Trainer optimization
trainer:
  accumulate_grad_batches: 2
  precision: "32-true"
  gradient_clip_val: 5.0
  max_epochs: 200
  benchmark: true
  accelerator: gpu
  devices: 1
  strategy: auto

# Dataloader optimization
dataloaders:
  train_dataloader:
    num_workers: 12
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 3
  val_dataloader:
    num_workers: 8
    pin_memory: true
    persistent_workers: true
    prefetch_factor: 2
  test_dataloader:
    num_workers: 4
    pin_memory: true
  predict_dataloader:
    num_workers: 4
    pin_memory: true

# Data Overrides
data:
  batch_size: 4
````

## File: model/presets/craft.yaml
````yaml
# @package model
# v5.0 | Purpose: Monolithic CRAFT Preset (Detection)
# Validated: 2026-01-18 | Deps: ocr.domains.detection
encoder:
  _target_: ocr.domains.detection.models.encoders.craft_vgg.CraftVGGEncoder
  model_name: "vgg16_bn"
  pretrained: true
  output_indices: [1, 2, 3, 4]
  extra_channels: 512
  freeze_backbone: false

decoder:
  _target_: ocr.domains.detection.models.decoders.craft_decoder.CraftDecoder
  in_channels: [64, 128, 256, 512]
  inner_channels: 256
  out_channels: 256
  use_batch_norm: true

head:
  _target_: ocr.domains.detection.models.heads.craft_head.CraftHead
  in_channels: 256
  hidden_channels: 128
  postprocess:
    text_threshold: 0.7
    link_threshold: 0.4
    low_text: 0.3
    min_area: 16
    expand_ratio: 1.5
````

## File: model/presets/dbnetpp.yaml
````yaml
# @package model
# v5.0 | Purpose: Monolithic DBNet++ Preset (Detection)
# Validated: 2026-01-18 | Deps: ocr.domains.detection
encoder:
  _target_: ocr.core.models.encoder.timm_backbone.TimmBackbone
  model_name: "resnet50"
  pretrained: true
  output_indices: [1, 2, 3, 4]
  in_channels: 3

decoder:
  _target_: ocr.domains.detection.models.decoders.dbpp_decoder.DBPHeadDecoder
  in_channels: [64, 128, 256, 512]
  inner_channels: 256
  out_channels: 128  # from component_overrides
  use_batch_norm: true
  adaptive: true
  attention: true

head:
  _target_: ocr.domains.detection.models.heads.db_head.DBHead
  in_channels: 128  # matched with decoder out_channels
  upscale: 4        # from component_overrides already matching, but explicit
  binarization_threshold: 0.3
  expand_ratio: 1.5
````

## File: model/presets/parseq.yaml
````yaml
# @package model
# v5.0 | Purpose: Monolithic PARSeq Preset (Recognition)
# Validated: 2026-01-18 | Deps: ocr.domains.recognition

# 1. Architecture Definition
architecture:
  _target_: ocr.domains.recognition.models.PARSeq
  backbone:
    _target_: ocr.core.models.encoder.TimmBackbone
    model_name: resnet18
    pretrained: true
  decoder:
    _target_: ocr.domains.recognition.models.decoder.PARSeqDecoder
    d_model: 512
    nhead: 8
    num_layers: 12

# 2. Tokenizer (Moved to Domain Controller)
# tokenizer: null

# 3. Loss Configuration (Moved to Domain Controller)
# loss: null

# 4. Local Optimization Defaults (Moved to Train/Optimizer)
# optimizer: null
````

## File: train/callbacks/default.yaml
````yaml
# @package _group_
# This file defines the callbacks that will be instantiated by Hydra.
# The `defaults` section includes individual callback configurations.
# Custom callbacks that don't have separate files are defined below.

defaults:
  - model_checkpoint@_group_.model_checkpoint
  - early_stopping@_group_.early_stopping
  - rich_progress_bar@_group_.rich_progress_bar
  - wandb_image_logging@_group_.wandb_image_logging
  - wandb_completion@_group_.wandb_completion
  - performance_profiler@_group_.performance_profiler
  - metadata@_group_.metadata  # Checkpoint Catalog V2: automatic metadata generation
  - _self_
````

## File: train/callbacks/early_stopping.yaml
````yaml
# @package _group_
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.EarlyStopping.html

_target_: lightning.pytorch.callbacks.EarlyStopping
monitor: "val/hmean" # quantity to be monitored, must be specified !!!
min_delta: 0. # minimum change in the monitored quantity to qualify as an improvement
patience: 5 # number of checks with no improvement after which training will be stopped
verbose: False # verbosity mode
mode: "max" # "max" means higher metric value is better, can be also "min"
strict: True # whether to crash the training if monitor is not found in the validation metrics
check_finite: True # when set True, stops training when the monitor becomes NaN or infinite
stopping_threshold: null # stop training immediately once the monitored quantity reaches this threshold
divergence_threshold: null # stop training as soon as the monitored quantity becomes worse than this threshold
check_on_train_epoch_end: false # whether to run early stopping at the end of the training epoch
# log_rank_zero_only: False  # this keyword argument isn't available in stable version
````

## File: train/callbacks/metadata.yaml
````yaml
# @package _group_
# Checkpoint Metadata Generation Callback
#
# Generates .metadata.yaml files alongside checkpoints using the Checkpoint
# Catalog V2 schema. This enables fast catalog building (40-100x speedup)
# without loading PyTorch checkpoint files.
#
# Benefits:
#   - Fast catalog builds: ~10ms per checkpoint (vs 2-5 seconds)
#   - Zero training overhead: metadata generation is < 1ms
#   - Required metrics included: precision, recall, hmean, epoch
#
# Usage:
#   Add to your training config:
#     defaults:
#       - callbacks/metadata
#
#   Or enable via command line:
#     python runners/train.py +callbacks/metadata=default

_target_: ocr.core.lightning.callbacks.MetadataCallback
exp_name: ${exp_name}  # From main config
outputs_dir: ${paths.output_dir}  # From paths config
training_phase: "training"  # Can override for finetuning
````

## File: train/callbacks/model_checkpoint.yaml
````yaml
# @package _group_
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html
#
# Enhanced checkpoint naming scheme for better organization and clarity
# Structure: <index>/checkpoints/<type>-<epoch>_<step>_<metric>.ckpt
#
# Example checkpoint names:
#   - Epoch: epoch-03_step-000103.ckpt
#   - Last:  last.ckpt
#   - Best:  best-hmean-0.8920.ckpt
#
# Directory structure:
#   outputs/1/checkpoints/
#   outputs/2/checkpoints/
#   outputs/3/checkpoints/
#   (Index-based numbering for easy sorting and to avoid overlaps)

_target_: ocr.core.lightning.callbacks.unique_checkpoint.UniqueModelCheckpoint
dirpath: ${global.paths.checkpoint_dir} # directory to save the model file

# Experiment identification
experiment_tag: ${oc.env:EXPERIMENT_TAG,${exp_name}} # Unique experiment identifier
training_phase: "training" # Stage: training, validation, finetuning, etc.
add_timestamp: true # Add timestamp to directory structure for uniqueness

# Checkpoint naming template (used for type determination)
# The actual naming is handled by the callback using the hierarchical scheme
filename: "best" # Template for best checkpoints

# Monitoring and saving configuration
# BUG-20251116-001: Reduced checkpoint saving to prevent excessive disk usage
monitor: "val/hmean" # name of the logged metric which determines when model is improving
verbose: False # verbosity mode - set to False to reduce log spam (BUG-20251116-001)
save_last: True # save last checkpoint for resuming training
save_top_k: 1 # save only the best model to reduce disk usage (BUG-20251116-001)
mode: "max" # "max" means higher metric value is better, can be also "min"
auto_insert_metric_name: True # Add metric name and value to checkpoint filename
save_weights_only: False # if True, then only the model's weights will be saved
every_n_train_steps: null # number of training steps between checkpoints
train_time_interval: null # checkpoints are monitored at the specified time interval
every_n_epochs: 1 # number of epochs between checkpoints - ensure we save checkpoints every epoch
save_on_train_epoch_end: False # whether to run checkpointing at the end of the training epoch or the end of validation
````

## File: train/callbacks/model_summary.yaml
````yaml
# @package _group_
# https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.RichModelSummary.html

model_summary:
  _target_: lightning.pytorch.callbacks.RichModelSummary
  max_depth: 1 # the maximum depth of layer nesting that the summary will include
````

## File: train/callbacks/performance_profiler.yaml
````yaml
# @package _group_
# Performance Profiler Callback Configuration
# Tracks validation performance to identify bottlenecks (Phase 1.1)

_target_: ocr.core.lightning.callbacks.PerformanceProfilerCallback
enabled: false
log_interval: 10  # Log batch metrics every N batches
profile_memory: true  # Track GPU/CPU memory usage
verbose: false  # Set to true for console output during debugging
````

## File: train/callbacks/recognition_wandb.yaml
````yaml
# @package callbacks

recognition_wandb:
  _target_: ocr.features.recognition.callbacks.wandb_image_logging.RecognitionWandbImageLogger
  log_every_n_epochs: 1
  num_samples: 8
````

## File: train/callbacks/rich_progress_bar.yaml
````yaml
# @package _group_
# https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.callbacks.RichProgressBar.html

_target_: lightning.pytorch.callbacks.RichProgressBar
theme:
  _target_: lightning.pytorch.callbacks.progress.rich_progress.RichProgressBarTheme
  progress_bar: red
  progress_bar_finished: green
  progress_bar_pulse: yellow
  time: cyan
  processing_speed: cyan
  description: bold white
  metrics: white
refresh_rate: 2
````

## File: train/callbacks/wandb_completion.yaml
````yaml
# @package _group_
# WandB Completion Callback
# Automatically marks the run as finished in Weights & Biases

_target_: ocr.core.lightning.callbacks.wandb_completion.WandbCompletionCallback
````

## File: train/callbacks/wandb_image_logging.yaml
````yaml
# @package _group_
_target_: ocr.core.lightning.callbacks.wandb_image_logging.WandbImageLoggingCallback
log_every_n_epochs: 1
````

## File: train/logger/default.yaml
````yaml
# @package _group_
# Logger config

defaults:
  - wandb
  - csv
  - _self_
````

## File: train/logger/wandb.yaml
````yaml
# @package _group_
# Logger config
_target_: lightning.pytorch.loggers.WandbLogger
project: "receipt-text-recognition-ocr-project"
log_model: "all"
save_dir: "${global.paths.wandb}"

# Configure W&B to save offline data under the canonical outputs root
settings:
  offline: false  # Don't run in offline mode
  save_code: false  # Don't save code snapshots
  sync_dir: "${global.paths.wandb_sync_root}"  # Custom sync directory under outputs/


# Per batch image logging for error analysis
per_batch_image_logging:
  enabled: false
  recall_threshold: 0.4
  max_batches_per_epoch: 2
  max_images_per_batch: 4
  use_transformed_batch: true
  image_format: jpeg
  max_image_side: 640
````

## File: train/optimizer/adam.yaml
````yaml
# @package model.optimizer

_target_: torch.optim.Adam
lr: 0.001
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0.0001
````

## File: train/optimizer/adamw.yaml
````yaml
# @package model.optimizer

_target_: torch.optim.AdamW
lr: 0.0003
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0.01
````

## File: main.yaml
````yaml
# @package _global_
# v5.0 | Purpose: Entry Point / Root Configuration
# Validated: 2026-01-18 | Deps: global/paths, hardware/rtx3060, global/default
defaults:
  - _self_
  - global: default
  - hardware: rtx3060
  - domain: ???          # Must be provided (detection, recognition, kie)
  - /data/runtime/performance/none@runtime
  - experiment: null

# The 'mode' determines which orchestrator or pipeline script is called
mode: train              # Options: [train, eval, predict]

# Placeholders for domain-specific logic
# model: null
# data: null
# train: null
````

## File: README.md
````markdown
# Hydra Configuration Architecture

**AI-Optimized Documentation**: [`AgentQMS/standards/tier2-framework/hydra-configuration-architecture.yaml`](../AgentQMS/standards/tier2-framework/hydra-configuration-architecture.yaml)

**Legacy Human-Readable Guide**: [`__LEGACY__/README_20260108_deprecated.md`](__LEGACY__/README_20260108_deprecated.md)

---

## Quick Reference
(Need new documentation)
````
