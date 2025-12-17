# Testing Strategy: Pipeline Validation Guide

**Purpose**: Comprehensive testing strategy to validate data contracts and prevent regression bugs in the OCR pipeline.

**Audience**: Developers implementing new features, QA engineers, and CI/CD pipelines.

---

## 📋 Testing Pyramid

```
┌─────────────────────────────────┐
│ End-to-End Integration Tests    │ ← Catch contract violations
│ (test_collate_integration.py)   │
├─────────────────────────────────┤
│ Component Integration Tests     │ ← Validate component interactions
│ (test_transform_pipeline_*.py)  │
├─────────────────────────────────┤
│ Unit Tests with Contracts       │ ← Validate individual functions
│ (test_*_contracts.py)          │
├─────────────────────────────────┤
│ Shape/Type Validation Tests     │ ← Prevent shape mismatches
│ (test_*_shapes.py)             │
└─────────────────────────────────┘
```

---

## 🧪 Test Categories

### 1. Data Contract Validation Tests

**Purpose**: Ensure all pipeline stages maintain data contracts.

**Files**:
- `tests/ocr/datasets/test_polygon_filtering.py` - Polygon shape contracts
- `tests/integration/test_collate_integration.py` - End-to-end contracts
- `tests/ocr/datasets/test_transform_pipeline_contracts.py` - Transform contracts

**Coverage**:
- ✅ Input validation for each pipeline stage
- ✅ Output format verification
- ✅ Shape consistency across stages
- ✅ Type safety (numpy vs torch, PIL vs numpy)

### 2. Edge Case and Error Handling Tests

**Purpose**: Test system robustness with invalid or edge case inputs.

**Test Cases**:
- Empty polygon lists
- Images with no text regions
- Corrupted map files
- Variable batch sizes
- Extreme image dimensions
- Invalid polygon shapes

### 3. Performance Regression Tests

**Purpose**: Detect performance degradation from contract violations.

**Metrics**:
- Transform throughput (> 100 images/sec)
- Memory usage (< 8GB per GPU)
- Training stability (loss convergence)
- Inference latency (< 50ms per batch)

### 4. Inference Component Integration Tests

**Purpose**: Validate modular inference components and orchestrator pattern.

**Files**:
- `tests/unit/test_orchestrator.py` - InferenceOrchestrator coordination
- `tests/unit/test_model_manager.py` - Model lifecycle and caching
- `tests/unit/test_preprocessing_pipeline.py` - Image preprocessing
- `tests/unit/test_postprocessing_pipeline.py` - Prediction decoding
- `tests/unit/test_preview_generator.py` - Preview encoding
- `tests/unit/test_image_loader.py` - Image I/O and EXIF
- `tests/unit/test_coordinate_manager.py` - Coordinate transformations
- `tests/unit/test_preprocessing_metadata.py` - Metadata calculation

**Coverage**:
- ✅ Component initialization and configuration
- ✅ Data contract compliance between components
- ✅ Error handling and edge cases
- ✅ Backward compatibility with InferenceEngine
- ✅ Coordinate transformation accuracy (BUG-20251116-001 fix)

**Test Execution**:
```bash
# Run all inference component tests
python -m pytest tests/unit/test_orchestrator.py \
                 tests/unit/test_model_manager.py \
                 tests/unit/test_preprocessing_pipeline.py \
                 tests/unit/test_postprocessing_pipeline.py \
                 tests/unit/test_preview_generator.py \
                 tests/unit/test_image_loader.py \
                 tests/unit/test_coordinate_manager.py \
                 tests/unit/test_preprocessing_metadata.py -v

# Run orchestrator tests specifically
python -m pytest tests/unit/test_orchestrator.py -v

# Test backward compatibility
python -c "from ocr.inference.engine import InferenceEngine; e = InferenceEngine(); print('✓ Import OK')"
```

**Component Test Matrix**:

| Component | Unit Tests | Integration Tests | Contract Tests |
|-----------|-----------|------------------|----------------|
| InferenceOrchestrator | 10/10 ✅ | Delegates to components | ✅ Data flow |
| ModelManager | ✅ Passing | Checkpoint loading | ✅ Config bundle |
| PreprocessingPipeline | ✅ Passing | Transform + metadata | ✅ PreprocessingResult |
| PostprocessingPipeline | ✅ Passing | Decode + format | ✅ PostprocessingResult |
| PreviewGenerator | ✅ Passing | Encode + attach | ✅ Base64 output |
| ImageLoader | ✅ Passing | EXIF normalization | ✅ LoadedImage |
| CoordinateManager | ✅ Passing | Transform accuracy | ✅ Padding position |
| PreprocessingMetadata | ✅ Passing | Metadata creation | ✅ Dictionary format |

---

## 🔄 CI/CD Integration

### Pre-commit Hooks

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: validate-data-contracts
        name: Validate Data Contracts
        entry: python scripts/validate_pipeline_contracts.py
        language: system
        files: \.(py)$
        pass_filenames: false

      - id: run-contract-tests
        name: Run Contract Tests
        entry: python -m pytest tests/ -k "contract" -x
        language: system
        files: \.(py)$
        pass_filenames: false
```

### GitHub Actions Workflow

```yaml
# .github/workflows/pipeline-validation.yml
name: Pipeline Contract Validation

on:
  push:
    paths:
      - 'ocr/datasets/**'
      - 'ocr/models/**'
      - 'tests/**'

jobs:
  validate-contracts:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run contract tests
        run: |
          python -m pytest tests/ -k "contract" --tb=short
      - name: Validate data contracts
        run: |
          python scripts/validate_pipeline_contracts.py
```

---

## 🛠️ Testing Utilities

### Contract Validation Script

```python
# scripts/validate_pipeline_contracts.py
import torch
from ocr.datasets import OCRDataset, DBCollateFN
from ocr.datasets.transforms import DBTransforms

def validate_full_pipeline():
    """Validate complete pipeline maintains contracts."""
    # Create test dataset
    dataset = OCRDataset(...)

    # Test dataset contract
    sample = dataset[0]
    validate_dataset_contract(sample)

    # Test transform contract
    transformed = transforms(sample)
    validate_transform_contract(transformed)

    # Test collate contract
    batch = collate_fn([transformed])
    validate_collate_contract(batch)

    print("✅ All contracts validated")

if __name__ == "__main__":
    validate_full_pipeline()
```

### Shape Debugging Helper

```python
# debug/shape_debugger.py
def debug_tensor_shapes(data, prefix=""):
    """Print tensor shapes for debugging."""
    if isinstance(data, dict):
        for key, value in data.items():
            debug_tensor_shapes(value, f"{prefix}.{key}")
    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            debug_tensor_shapes(item, f"{prefix}[{i}]")
    elif hasattr(data, 'shape'):
        print(f"{prefix}: {data.shape}")
    else:
        print(f"{prefix}: {type(data)}")
```

---

## 📊 Test Coverage Metrics

### Required Coverage

- **Data Contract Tests**: > 95% of contract rules
- **Error Handling**: All documented error conditions
- **Edge Cases**: All boundary conditions
- **Integration Paths**: All pipeline combinations

### Coverage Report

```bash
# Generate coverage report
python -m pytest tests/ --cov=ocr/ --cov-report=html

# Check contract-specific coverage
python -m pytest tests/ -k "contract" --cov=ocr/ --cov-report=term-missing
```

---

## 🚨 Failure Analysis

### Common Test Failures

1. **Shape Mismatch Errors**
   ```
   ValueError: Target size (2, 1, 224, 224) must be the same as input size (2, 1, 896, 896)
   ```
   **Cause**: Head upscale factor not matched in ground truth
   **Fix**: Update ground truth dimensions: `gt = F.interpolate(gt, scale_factor=4)`

2. **Channel Count Errors**
   ```
   RuntimeError: expected input[2, 3, 224, 224] to have 256 channels, but got 3
   ```
   **Cause**: Raw images passed to head instead of decoder features
   **Fix**: Use full pipeline or mock decoder output

3. **Type Errors**
   ```
   AttributeError: 'Image' object has no attribute 'shape'
   ```
   **Cause**: PIL Image where numpy array expected
   **Fix**: Convert with `np.array(pil_image)`

### Debugging Workflow

1. **Identify Failure Point**
   - Run with `--tb=long` for full traceback
   - Check tensor shapes: `print(x.shape for x in [input1, input2, ...])`

2. **Validate Contracts**
   - Run `scripts/validate_pipeline_contracts.py`
   - Check against `docs/pipeline/data_contracts.md`

3. **Isolate Component**
   - Test individual pipeline stages
   - Use mock data to isolate issues

4. **Fix and Retest**
   - Apply contract-compliant fix
   - Run full test suite
   - Update contracts if needed

---

## 📈 Continuous Improvement

### Test Evolution

- **Add tests** for new features that affect data flow
- **Update contracts** when data formats change
- **Expand coverage** for uncovered edge cases
- **Performance benchmarks** for regression detection

### Quality Gates

- [ ] All contract tests pass
- [ ] No shape-related runtime errors
- [ ] Performance within acceptable ranges
- [ ] Documentation updated for changes

---

**Last Updated**: December 15, 2025
**Version**: 1.1 (Added inference component integration tests)
