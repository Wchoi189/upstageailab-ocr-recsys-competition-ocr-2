---
title: "2025 11 12 Plan 004 Revised Inference Consolidation"
date: "2025-12-06 18:08 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---



# PLAN-004 REVISED: Minimal Inference Service Consolidation

**Created**: 2025-11-12
**Original Plan**: 2025-11-11_plan-004-inference-service-consolidation.md
**Status**: APPROVED (Revised - Low Risk)
**Priority**: Medium
**Risk Level**: LOW (reduced from VERY HIGH)
**Estimated Effort**: 1-2 weeks (reduced from 4-5 weeks)
**Success Likelihood**: 85% (increased from 50%)

---

## Executive Summary

This is a **revised, safer version** of PLAN-004 that achieves 80% of the benefits with 20% of the risk. Instead of implementing complex checkpoint caching and singleton patterns, we focus on:

1. **Eliminating tempfile overhead** (direct numpy array passing)
2. **Deduplicating service code** (shared base class)
3. **Adding observability** (metrics for future optimization)

**Deferred until proven necessary**:
- ❌ In-memory checkpoint caching (memory leak risk)
- ❌ Singleton pattern (thread-safety complexity)
- ❌ Complex engine lifecycle management

---

## Problem Statement

### Current Architecture Issues

```
InferenceRunner (ui/apps/inference/services/)
├── Creates numpy array from image
├── Writes to tempfile.NamedTemporaryFile
├── Passes file path to InferenceEngine
└── InferenceEngine reads file back to numpy

InferenceService (ui/apps/unified_ocr_app/services/)
├── Creates numpy array from image
├── Writes to tempfile.NamedTemporaryFile  [DUPLICATE CODE]
├── Passes file path to InferenceEngine    [DUPLICATE CODE]
└── InferenceEngine reads file back to numpy
```

**Problems**:
1. **Tempfile Overhead**: Unnecessary disk I/O for in-memory data
2. **Code Duplication**: Two services with identical logic
3. **No Observability**: Can't measure performance bottlenecks

---

## Goals & Non-Goals

### Goals ✅
- ✅ Eliminate tempfile overhead (direct numpy passing)
- ✅ Deduplicate service code (shared base class)
- ✅ Add metrics for inference performance
- ✅ Maintain backward compatibility
- ✅ Keep implementation simple and testable

### Non-Goals ❌
- ❌ In-memory checkpoint caching (too risky)
- ❌ Singleton engine pattern (thread-safety issues)
- ❌ Complex lifecycle management
- ❌ API-breaking changes

---

## Implementation Plan

### Phase 1: Investigation & Metrics (2 days)

**Goal**: Understand current state before making changes

**Tasks**:

1. **Service Comparison**
   ```bash
   # Compare the two services
   diff -u ui/apps/inference/services/inference_runner.py \
           ui/apps/unified_ocr_app/services/inference_service.py

   # Check for actual usage
   grep -r "InferenceRunner\|InferenceService" ui/apps/
   ```

2. **Add Performance Metrics**
   ```python
   # Add to InferenceEngine
   import time
   import logging

   logger = logging.getLogger(__name__)

   class InferenceMetrics:
       """Track inference performance metrics"""
       inference_count = 0
       total_inference_time = 0.0
       model_load_count = 0
       total_load_time = 0.0

       @classmethod
       def log_summary(cls):
           if cls.inference_count > 0:
               avg_inference = cls.total_inference_time / cls.inference_count
               logger.info(
                   f"Inference metrics: "
                   f"count={cls.inference_count}, "
                   f"avg_time={avg_inference:.3f}s, "
                   f"model_loads={cls.model_load_count}"
               )
   ```

3. **Identify Tempfile Usage**
   ```bash
   # Find all tempfile usage
   grep -rn "tempfile.NamedTemporaryFile" ui/apps/

   # Check if used elsewhere
   grep -rn "\.name\|\.file" ui/apps/ | grep -i temp
   ```

**Deliverables**:
- Service comparison report
- Metrics baseline (current performance)
- Tempfile usage inventory

**Exit Criteria**:
- Understand if services are true duplicates
- Have baseline performance numbers
- Know all tempfile usage locations

---

### Phase 2: Eliminate Tempfile Overhead (3 days)

**Goal**: Enable direct numpy array passing

**Implementation**:

```python
# ocr/inference/engine.py (or wherever InferenceEngine lives)

class InferenceEngine:
    def predict(self, input_data: Union[np.ndarray, str, Path]):
        """Accept numpy arrays OR file paths.

        Args:
            input_data: Either numpy array (new) or file path (legacy)

        Returns:
            Prediction results
        """
        if isinstance(input_data, np.ndarray):
            # NEW: Direct numpy array path (no disk I/O)
            return self._predict_from_array(input_data)
        elif isinstance(input_data, (str, Path)):
            # LEGACY: File path support (backward compatible)
            return self._predict_from_file(input_data)
        else:
            raise TypeError(
                f"input_data must be np.ndarray or path, got {type(input_data)}"
            )

    def _predict_from_array(self, image_array: np.ndarray) -> dict:
        """New optimized path: direct numpy array processing"""
        start_time = time.time()

        # Process directly without tempfile
        results = self._run_inference(image_array)

        InferenceMetrics.inference_count += 1
        InferenceMetrics.total_inference_time += time.time() - start_time

        return results

    def _predict_from_file(self, image_path: Union[str, Path]) -> dict:
        """Legacy path: load from file (kept for compatibility)"""
        image_array = self._load_image(image_path)
        return self._predict_from_array(image_array)
```

**Testing**:
```python
# tests/unit/test_inference_engine.py

def test_predict_accepts_numpy_array():
    """Test new numpy array path"""
    engine = InferenceEngine()
    image_array = np.random.rand(480, 640, 3).astype(np.float32)

    result = engine.predict(image_array)

    assert result is not None
    assert "boxes" in result

def test_predict_accepts_file_path():
    """Test legacy file path still works"""
    engine = InferenceEngine()

    result = engine.predict("/path/to/image.jpg")

    assert result is not None
    assert "boxes" in result
```

**Deliverables**:
- InferenceEngine accepts numpy arrays
- Unit tests for both paths
- Backward compatibility maintained

**Exit Criteria**:
- ✅ All existing tests pass
- ✅ New numpy array tests pass
- ✅ No API-breaking changes

---

### Phase 3: Deduplicate Service Code (2 days)

**Goal**: Create shared base class for both services

**Implementation**:

```python
# ui/apps/shared/inference_base.py (new file)

from abc import ABC
import numpy as np
from ocr.inference.engine import InferenceEngine

class BaseInferenceService(ABC):
    """Shared inference service logic.

    Both InferenceRunner and InferenceService inherit from this
    to eliminate code duplication.
    """

    def __init__(self, config=None):
        self.config = config
        self._engine = None

    def infer(self, image_array: np.ndarray) -> dict:
        """Run inference on image array.

        Args:
            image_array: Image as numpy array (H, W, 3)

        Returns:
            Prediction results with bounding boxes
        """
        engine = self._get_engine()

        # NEW: Pass numpy array directly (no tempfile)
        return engine.predict(image_array)

    def _get_engine(self) -> InferenceEngine:
        """Get or create inference engine.

        Simple approach: Create new engine each time.
        No caching, no singleton, no complexity.
        """
        if self._engine is None:
            self._engine = InferenceEngine(config=self.config)
        return self._engine
```

**Update Existing Services**:

```python
# ui/apps/inference/services/inference_runner.py

from ui.apps.shared.inference_base import BaseInferenceService

class InferenceRunner(BaseInferenceService):
    """Inference service for standalone inference app.

    Inherits all logic from BaseInferenceService.
    Add app-specific customizations here if needed.
    """
    pass  # Just inherits, no duplication


# ui/apps/unified_ocr_app/services/inference_service.py

from ui.apps.shared.inference_base import BaseInferenceService

class InferenceService(BaseInferenceService):
    """Inference service for unified OCR app.

    Inherits all logic from BaseInferenceService.
    Add app-specific customizations here if needed.
    """
    pass  # Just inherits, no duplication
```

**Migration Strategy**:
1. Create `BaseInferenceService` with shared logic
2. Update `InferenceRunner` to inherit from it
3. Run tests for inference app
4. Update `InferenceService` to inherit from it
5. Run tests for unified OCR app
6. Remove old duplicated code

**Deliverables**:
- `BaseInferenceService` base class
- Both services inheriting from base
- All service tests passing

**Exit Criteria**:
- ✅ No code duplication between services
- ✅ All app tests pass
- ✅ Services maintain same behavior

---

### Phase 4: Observability & Monitoring (1 day)

**Goal**: Add metrics to measure improvements

**Implementation**:

```python
# Add to InferenceEngine or BaseInferenceService

class InferenceMetrics:
    """Track inference performance metrics"""

    # Counters
    inference_count = 0
    tempfile_usage_count = 0  # Should go to 0

    # Timings
    total_inference_time = 0.0
    total_load_time = 0.0

    # Tracking
    checkpoint_loads = {}  # Track how often each checkpoint is loaded

    @classmethod
    def track_inference(cls, duration: float):
        cls.inference_count += 1
        cls.total_inference_time += duration

    @classmethod
    def track_checkpoint_load(cls, checkpoint_path: str, duration: float):
        if checkpoint_path not in cls.checkpoint_loads:
            cls.checkpoint_loads[checkpoint_path] = {"count": 0, "total_time": 0.0}
        cls.checkpoint_loads[checkpoint_path]["count"] += 1
        cls.checkpoint_loads[checkpoint_path]["total_time"] += duration

    @classmethod
    def get_summary(cls) -> dict:
        """Get metrics summary"""
        return {
            "inference_count": cls.inference_count,
            "avg_inference_time": (
                cls.total_inference_time / cls.inference_count
                if cls.inference_count > 0 else 0
            ),
            "tempfile_usage_count": cls.tempfile_usage_count,
            "checkpoint_loads": cls.checkpoint_loads,
        }

    @classmethod
    def log_summary(cls):
        """Log metrics summary"""
        summary = cls.get_summary()
        logger.info(f"Inference metrics: {summary}")
```

**Integration**:
```python
# In InferenceEngine._predict_from_array
start = time.time()
results = self._run_inference(image_array)
InferenceMetrics.track_inference(time.time() - start)

# In InferenceEngine.load_checkpoint
start = time.time()
model = self._load_model(checkpoint_path)
InferenceMetrics.track_checkpoint_load(checkpoint_path, time.time() - start)
```

**Deliverables**:
- Metrics collection in InferenceEngine
- Summary logging at application exit
- Dashboard-ready metrics export

**Exit Criteria**:
- ✅ Can measure inference latency
- ✅ Can track checkpoint load frequency
- ✅ Tempfile usage counter at 0

---

## Risk Mitigation

### Low-Risk Approach

| Risk Factor | Original Plan | Revised Plan | Mitigation |
|-------------|---------------|--------------|------------|
| Memory leaks | HIGH (in-memory cache) | LOW (no cache) | Don't cache models in memory |
| Thread safety | HIGH (singleton) | NONE (no singleton) | Each request gets own engine |
| API changes | MEDIUM | NONE | Backward compatible |
| Testing complexity | HIGH | LOW | Simple, testable changes |
| Rollback difficulty | HIGH | LOW | Feature flag for numpy path |

### Feature Flags

```python
# Enable/disable numpy array optimization
USE_NUMPY_DIRECT = os.getenv("INFERENCE_USE_NUMPY_DIRECT", "true").lower() == "true"

class InferenceEngine:
    def predict(self, input_data):
        if USE_NUMPY_DIRECT and isinstance(input_data, np.ndarray):
            return self._predict_from_array(input_data)
        # ... fallback to tempfile path
```

### Rollback Plan

1. Set `INFERENCE_USE_NUMPY_DIRECT=false` to revert to tempfile path
2. Restore old service files from git
3. No database or checkpoint changes needed

---

## Testing Strategy

### Unit Tests
```python
# Test numpy array path
def test_engine_predict_numpy_array()
def test_engine_predict_file_path()
def test_service_infer_with_array()

# Test metrics
def test_metrics_tracking()
def test_checkpoint_load_tracking()
```

### Integration Tests
```python
# Test both services still work
def test_inference_runner_full_flow()
def test_inference_service_full_flow()

# Test performance improvement
def test_numpy_path_faster_than_tempfile()
```

### Performance Tests
```bash
# Before/after comparison
python scripts/benchmark_inference.py --mode=tempfile
python scripts/benchmark_inference.py --mode=numpy

# Expected: 10-30% latency reduction from eliminating disk I/O
```

---

## Success Metrics

### Performance Targets
- **Inference latency**: Reduce p50 by >10%
- **Tempfile usage**: Reduce to 0 calls
- **Code duplication**: Reduce by 100% (shared base class)

### Quality Targets
- **Test coverage**: 100% for new code
- **Production incidents**: 0 in first 2 weeks
- **Rollback rate**: <5%

### Monitoring
```python
# Track in production
metrics = {
    "inference_latency_p50": histogram,
    "inference_latency_p95": histogram,
    "tempfile_usage_count": counter,
    "checkpoint_load_frequency": counter,
    "memory_usage": gauge,
}
```

---

## Implementation Timeline

| Phase | Duration | Effort | Dependencies |
|-------|----------|--------|--------------|
| 1. Investigation | 2 days | 0.5 days hands-on | None |
| 2. Numpy Arrays | 3 days | 2 days hands-on | Phase 1 |
| 3. Deduplicate | 2 days | 1.5 days hands-on | Phase 2 |
| 4. Observability | 1 day | 0.5 days hands-on | Phase 3 |
| **Total** | **8 days** | **4.5 days** | - |

**Calendar Time**: 2 weeks (with testing & review)

---

## Decision: Why This Approach?

### Comparison with Original Plan

| Aspect | Original PLAN-004 | Revised Plan | Winner |
|--------|-------------------|--------------|--------|
| Risk | Very High | Low | ✅ Revised |
| Complexity | 7 phases | 4 phases | ✅ Revised |
| Duration | 4-5 weeks | 2 weeks | ✅ Revised |
| Benefits | 100% | 80% | Original |
| Success likelihood | 50% | 85% | ✅ Revised |
| Thread safety | Complex | N/A | ✅ Revised |
| Memory leaks | Risk | No risk | ✅ Revised |

### What We're NOT Doing (And Why)

1. **Checkpoint Caching**:
   - ❌ Original plan had in-memory cache
   - ✅ Revised: Let PyTorch handle it
   - **Reason**: Memory leak risk not worth unproven benefit

2. **Singleton Pattern**:
   - ❌ Original plan had shared engine instance
   - ✅ Revised: Per-request engines
   - **Reason**: Thread-safety complexity

3. **Complex Lifecycle**:
   - ❌ Original plan had LRU eviction
   - ✅ Revised: Simple create/destroy
   - **Reason**: YAGNI (You Aren't Gonna Need It)

---

## Future Enhancements (Phase 2)

**ONLY implement if metrics show need**:

1. **Simple Checkpoint Cache** (if same checkpoint loaded >5x/session)
   ```python
   # Simple dict cache (no LRU, no threading)
   _checkpoint_cache = {}

   def load_checkpoint(path):
       if path not in _checkpoint_cache:
           _checkpoint_cache[path] = _load_from_disk(path)
       return _checkpoint_cache[path]
   ```

2. **Connection Pooling** (if engine creation is slow)
   ```python
   # Simple pool (no singleton)
   engine_pool = [InferenceEngine() for _ in range(3)]

   def get_engine():
       return engine_pool[random.randint(0, 2)]
   ```

**Decision Point**: Only add if:
- Profiling shows clear bottleneck
- Benefits outweigh complexity
- Can maintain low risk profile

---

## Appendix: Investigation Checklist

### Before Starting Implementation

- [ ] Compare both service files (diff)
- [ ] Check if both are actively used
- [ ] Measure current inference latency
- [ ] Count tempfile operations
- [ ] Check thread usage (concurrent requests?)
- [ ] Profile model loading time
- [ ] Measure memory usage baseline
- [ ] Review existing tests

### Success Criteria

- [ ] Tempfile usage = 0
- [ ] Code duplication = 0
- [ ] Inference latency reduced >10%
- [ ] All tests passing
- [ ] No production incidents
- [ ] Memory usage stable
- [ ] Rollback plan tested

---

## Sign-off

**Approved by**: Implementation Team
**Date**: 2025-11-12
**Risk Assessment**: LOW (reduced from VERY HIGH)
**Go/No-Go**: ✅ GO

**Next Steps**:
1. Begin Phase 1 investigation
2. Collect baseline metrics
3. Create feature branch: `feature/inference-consolidation-minimal`
4. Implement phases sequentially
5. Monitor production metrics for 2 weeks post-deployment
