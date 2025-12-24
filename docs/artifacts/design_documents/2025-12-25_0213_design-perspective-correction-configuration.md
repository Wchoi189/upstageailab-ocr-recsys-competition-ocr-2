---
ads_version: "1.0"
type: "design"
category: "architecture"
status: "active"
version: "1.0"
tags: ['design', 'architecture', 'specification']
title: "Perspective Correction Configuration Reference"
date: "2025-12-25 02:13 (KST)"
branch: "main"
---

# Perspective Correction Configuration Reference

## Overview

This document provides the design and configuration reference for integrating perspective correction and image enhancement preprocessing into the OCR inference pipeline.

## Problem Statement

OCR accuracy suffers on images with perspective distortion, background tints, or poor contrast. The system needs flexible preprocessing options that can be enabled/disabled without breaking existing workflows.

## Design Goals

- Provide comprehensive preprocessing options for image enhancement
- Maintain backward compatibility with existing configurations
- Support both configuration-based and API-based enablement
- Enable easy testing and validation of preprocessing effects
- Ensure safe defaults (all enhancements disabled by default)

## Architecture

### High-Level Architecture

```
Inference Pipeline
├── Input Image
├── Preprocessing Pipeline
│   ├── Perspective Correction (rembg + 4-point transform)
│   ├── Background Normalization (gray-world method)
│   ├── Sepia Enhancement
│   ├── Grayscale Conversion
│   └── CLAHE Enhancement
├── OCR Model
└── Output Predictions
```

### Components

#### Configuration System
- **Purpose**: Centralized configuration management for preprocessing options
- **Responsibilities**: Define default values, validate configurations, support overrides
- **Interfaces**: YAML config files, command-line arguments, Python API parameters

#### Preprocessing Pipeline
- **Purpose**: Apply image enhancements in sequence
- **Responsibilities**: Execute enabled preprocessing steps, maintain image quality
- **Interfaces**: Receives image and config, returns processed image

#### Inference Engine
- **Purpose**: Main OCR prediction interface
- **Responsibilities**: Load models, run predictions, handle preprocessing integration
- **Interfaces**: Python API with preprocessing parameters

#### Test Scripts
- **Purpose**: Validate preprocessing integration and effects
- **Responsibilities**: Generate comparison visualizations, test different configurations
- **Interfaces**: Command-line interface for testing

## Design Decisions

### Configuration Structure
- **Decision**: Use hierarchical YAML configuration with `preprocessing` section
- **Rationale**: Clear organization, easy to extend, consistent with existing config patterns
- **Alternatives Considered**: Flat config, separate config files
- **Trade-offs**: Slightly more complex nesting vs. better organization

### Default Values
- **Decision**: All preprocessing options disabled by default
- **Rationale**: Backward compatibility, safe deployment
- **Alternatives Considered**: Enable commonly used options
- **Trade-offs**: More configuration required vs. potential unexpected behavior

### API Integration
- **Decision**: Add preprocessing parameters directly to `predict_image()` method
- **Rationale**: Seamless integration, no breaking changes to existing API
- **Alternatives Considered**: Separate preprocessing API
- **Trade-offs**: Larger method signature vs. cleaner separation

## Configuration Options

### Perspective Correction
```yaml
preprocessing:
  enable_perspective_correction: true
  perspective_display_mode: "corrected"  # or "original"
```

- **enable_perspective_correction**: Uses rembg for background removal, mask-based rectangle fitting, and 4-point transform
- **perspective_display_mode**:
  - `"corrected"`: Display results on corrected image (default)
  - `"original"`: Transform polygons back to original image space

### Full Enhancement Pipeline
```yaml
preprocessing:
  enable_perspective_correction: true
  perspective_display_mode: "corrected"
  enable_background_normalization: true  # Gray-world method
  enable_sepia_enhancement: false
  enable_grayscale: false
  enable_clahe: false
```

## Implementation Details

### Files Modified

1. `configs/_base/preprocessing.yaml` - Added all preprocessing options
2. `configs/predict.yaml` - Included preprocessing config + override examples
3. `configs/examples/predict_with_perspective.yaml` - New example config
4. `configs/examples/predict_full_enhancement.yaml` - New example config
5. `configs/examples/README.md` - Complete usage guide
6. `scripts/test_perspective_inference.py` - Validation test script

### Usage Examples

#### Method 1: Using Example Configs
```bash
# With perspective correction only
uv run python runners/predict.py \
  --config-name predict_with_perspective \
  checkpoint_path=outputs/.../checkpoint.ckpt \
  image_dir=data/test_images

# With full enhancement pipeline
uv run python runners/predict.py \
  --config-name predict_full_enhancement \
  checkpoint_path=outputs/.../checkpoint.ckpt \
  image_dir=data/test_images
```

#### Method 2: Command-Line Override
```bash
uv run python runners/predict.py \
  --config-name predict \
  checkpoint_path=outputs/.../checkpoint.ckpt \
  image_dir=data/test_images \
  preprocessing.enable_perspective_correction=true \
  preprocessing.enable_background_normalization=true
```

#### Method 3: Direct API (Python)
```python
from ocr.inference.engine import InferenceEngine

engine = InferenceEngine()
engine.load_model(checkpoint_path="...", config_path="...")

# Enable perspective correction
predictions = engine.predict_image(
    image_path="test.jpg",
    enable_perspective_correction=True,
    enable_background_normalization=True,
    perspective_display_mode="corrected"
)
```

#### Method 4: Test Script
```bash
# Test on single image with visualizations
python scripts/test_perspective_inference.py \
  --image data/test_images/sample.jpg \
  --checkpoint outputs/.../checkpoint.ckpt \
  --output outputs/perspective_test
```

## Validation and Testing

### Experiment Validation
Based on: `experiment-tracker/experiments/20251217_024343_image_enhancements_implementation`

#### Results Summary
- **Perspective Correction**: ✅ One extreme case: -83° → 0.88° skew correction
- **Background Normalization**: ✅ 75% tint reduction, 100% success rate (6/6 images)
- **Sepia Enhancement**: ⚠️ Under testing as alternative
- **Deskewing**: ❌ Excluded from integration (no OCR improvement)

### Recommended Configuration
```yaml
preprocessing:
  enable_perspective_correction: true
  enable_background_normalization: true
  enable_sepia_enhancement: false
```

## Next Steps

1. **Test the integration**:
   ```bash
   python scripts/test_perspective_inference.py \
     --image data/test_images/sample.jpg \
     --output outputs/test
   ```

2. **Run on worst performers**:
   ```bash
   uv run python runners/predict.py \
     --config-name predict_full_enhancement \
     checkpoint_path=outputs/.../checkpoint.ckpt \
     image_dir=data/zero_prediction_worst_performers
   ```

3. **Enable in UI**: The UI configs already have perspective correction fields

4. **Benchmark performance**: Compare OCR accuracy with/without perspective correction

## Related Files

- **Core Implementation**: `ocr/utils/perspective_correction/`
- **Preprocessing Pipeline**: `ocr/inference/preprocessing_pipeline.py`
- **Inference Engine**: `ocr/inference/engine.py`
- **Experiment Scripts**: `experiment-tracker/experiments/20251217_024343_image_enhancements_implementation/scripts/`

### Decision 1
- **Context**: Why this decision was needed
- **Options Considered**: Alternative approaches
- **Decision**: What was chosen
- **Rationale**: Why this option was selected
- **Consequences**: Implications of this choice

### Decision 2
- **Context**: Why this decision was needed
- **Options Considered**: Alternative approaches
- **Decision**: What was chosen
- **Rationale**: Why this option was selected
- **Consequences**: Implications of this choice

## Implementation Considerations

### Technical Requirements
- Requirement 1
- Requirement 2

### Dependencies
- Dependency 1
- Dependency 2

### Constraints
- Constraint 1
- Constraint 2

## Testing Strategy

### Unit Testing
- Test approach for individual components

### Integration Testing
- Test approach for component interactions

### End-to-End Testing
- Test approach for complete workflows

## Deployment

### Deployment Strategy
- How this will be deployed

### Rollback Plan
- How to rollback if issues occur

## Monitoring & Observability

### Metrics
- Key metrics to monitor

### Logging
- Logging strategy

### Alerting
- Alert conditions and thresholds

## Future Considerations

### Scalability
- How this design will scale

### Extensibility
- How this design can be extended

### Maintenance
- Maintenance considerations

---

*This design document follows the project's standardized format for architectural documentation.*
