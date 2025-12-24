---
ads_version: "1.0"
title: "Phase 1: Core Recognition Implementation"
date: "2025-12-25 04:10 (KST)"
type: "implementation_plan"
category: "text_recognition"
status: "active"
version: "1.0"
tags: ['implementation', 'recognition', 'paddleocr', 'phase1']
depends_on: []
successor: "2025-12-25_0410_implementation_plan_phase2-pipeline-integration.md"
---

# Phase 1: Core Recognition Implementation Plan

## Goal Description

Replace the current `StubRecognizer` with a production-ready `PaddleOCRRecognizer` backend using PaddleOCR PP-OCRv5. This phase establishes the foundation for text recognition without modifying the main inference pipeline, ensuring zero risk to existing detection functionality.

**Scope**: PaddleOCR integration only. No pipeline changes, no VLM, no extraction integration.

**Timeline**: 7-10 working days

**Success Criteria**:
- `PaddleOCRRecognizer` passes all 40 existing `test_recognizer_contract.py` tests
- Batch processing handles 100+ crops without OOM on RTX 3090
- VRAM usage ≤ 1GB for recognition model
- Inference speed ≤ 400ms for batch of 32 crops

---

## User Review Required

> [!IMPORTANT]
> **Decision Point**: PaddleOCR requires PaddlePaddle framework installation. This adds ~2GB to the Docker image and introduces a new dependency. Confirm this is acceptable before proceeding.

> [!WARNING]
> **Breaking Change**: The `RecognizerBackend` enum will be extended with a new `PADDLEOCR` value. Existing Hydra configs referencing `backend: stub` will continue to work.

---

## Proposed Changes

### OCR Inference Module

#### [NEW] `ocr/inference/backends/__init__.py`
Create backends subpackage for recognizer implementations.

#### [NEW] `ocr/inference/backends/paddleocr_recognizer.py`
Implement `PaddleOCRRecognizer` class extending `BaseRecognizer`.

```python
class PaddleOCRRecognizer(BaseRecognizer):
    """PaddleOCR PP-OCRv5 recognition backend."""

    def __init__(self, config: RecognizerConfig):
        self._config = config
        self._ocr = PaddleOCR(
            use_gpu=True,
            lang=config.language,
            rec_model_dir=config.model_path,
            use_angle_cls=False,  # Already handled by CropExtractor
            det=False,  # Recognition only
        )

    def recognize_batch(self, inputs: list[RecognitionInput]) -> list[RecognitionOutput]:
        crops = [inp.crop for inp in inputs]
        results = self._ocr.ocr(crops, det=False, cls=False)
        return [
            RecognitionOutput(
                text=res[0][1][0] if res else "",
                confidence=res[0][1][1] if res else 0.0,
            )
            for res in results
        ]
```

---

#### [MODIFY] `ocr/inference/recognizer.py`

Add PaddleOCR backend to the factory method.

```diff
def _create_backend(self) -> BaseRecognizer:
    if self.config.backend == RecognizerBackend.STUB:
        return StubRecognizer(self.config)
+   elif self.config.backend == RecognizerBackend.PADDLEOCR:
+       from .backends.paddleocr_recognizer import PaddleOCRRecognizer
+       return PaddleOCRRecognizer(self.config)
    elif self.config.backend == RecognizerBackend.TROCR:
        raise NotImplementedError("TrOCR backend not yet implemented.")
```

---

### Configuration

#### [NEW] `configs/recognition/paddleocr.yaml`
Create PaddleOCR-specific configuration.

```yaml
defaults:
  - default
  - _self_

recognition:
  enabled: true
  backend: "paddleocr"
  model_path: null  # Uses default PP-OCRv5 server model
  max_batch_size: 32
  target_height: 48  # PP-OCRv5 optimal height
  language: "korean"
  device: "cuda"

# PaddleOCR-specific settings
paddleocr:
  use_gpu: true
  use_angle_cls: false
  use_space_char: true
  rec_algorithm: "SVTR_LCNet"
  rec_char_dict_path: null  # Uses built-in Korean dictionary
```

---

### Dependencies

#### [MODIFY] `pyproject.toml`

Add PaddlePaddle and PaddleOCR dependencies.

```diff
[project.optional-dependencies]
recognition = [
+   "paddlepaddle-gpu>=3.0.0b1",
+   "paddleocr>=2.9.0",
]
```

---

### Tests

#### [NEW] `tests/unit/test_paddleocr_recognizer.py`
Backend-specific unit tests.

```python
class TestPaddleOCRRecognizer:
    """Tests for PaddleOCR recognition backend."""

    def test_initialization(self):
        """Test recognizer initializes with GPU."""

    def test_single_crop_recognition(self):
        """Test recognition of single Korean text crop."""

    def test_batch_recognition(self):
        """Test batch recognition of 32 crops."""

    def test_empty_crop_handling(self):
        """Test graceful handling of empty/invalid crops."""

    def test_vram_usage(self):
        """Verify VRAM usage stays under 1GB."""
```

#### [MODIFY] `tests/unit/test_recognizer_contract.py`
Add parametrized tests for PaddleOCR backend.

```diff
@pytest.fixture(params=["stub", "paddleocr"])
def recognizer_backend(request):
    """Parametrize tests across all backends."""
    return request.param
```

---

### CI/CD

#### [MODIFY] `.github/workflows/ci.yml`

Add recognition test job with GPU runner.

```diff
jobs:
  build-and-test:
    # ... existing config ...

+ recognition-tests:
+   runs-on: self-hosted  # GPU runner
+   needs: build-and-test
+   steps:
+     - uses: actions/checkout@v4
+     - name: Install recognition dependencies
+       run: uv sync --extra recognition
+     - name: Run recognition tests
+       run: uv run pytest tests/unit/test_*recognizer*.py -v
```

---

## Verification Plan

### Automated Tests

```bash
# 1. Run existing contract tests (must pass 100%)
uv run pytest tests/unit/test_recognizer_contract.py -v

# 2. Run new PaddleOCR-specific tests
uv run pytest tests/unit/test_paddleocr_recognizer.py -v

# 3. Memory profiling (must stay under 1GB)
uv run python -c "
from ocr.inference.recognizer import TextRecognizer, RecognizerConfig, RecognizerBackend
import torch
config = RecognizerConfig(backend=RecognizerBackend.PADDLEOCR)
recognizer = TextRecognizer(config=config)
print(f'VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
"

# 4. Throughput benchmark (must be ≤400ms for batch=32)
uv run python scripts/benchmark_recognition.py --backend paddleocr --batch-size 32
```

### Manual Verification

1. **Visual Inspection**: Run recognition on 10 sample receipts, verify Korean text accuracy
2. **VRAM Monitoring**: Use `nvidia-smi` during batch processing
3. **Error Handling**: Test with corrupted/blank images

---

## Task Checklist

- [ ] **Day 1: Environment Setup**
  - [ ] Install PaddlePaddle GPU
  - [ ] Install PaddleOCR
  - [ ] Verify GPU detection
  - [ ] Download PP-OCRv5 server model

- [ ] **Day 2-3: Core Implementation**
  - [ ] Create `backends/` subpackage
  - [ ] Implement `PaddleOCRRecognizer` class
  - [ ] Implement `recognize_single()` method
  - [ ] Implement `recognize_batch()` method
  - [ ] Implement `is_loaded()` method

- [ ] **Day 4: Factory Integration**
  - [ ] Update `RecognizerBackend` enum
  - [ ] Update `TextRecognizer._create_backend()`
  - [ ] Create `configs/recognition/paddleocr.yaml`

- [ ] **Day 5-6: Testing**
  - [ ] Write `test_paddleocr_recognizer.py`
  - [ ] Parametrize `test_recognizer_contract.py`
  - [ ] Run full test suite
  - [ ] Fix any failures

- [ ] **Day 7: Benchmarking & Documentation**
  - [ ] Create `scripts/benchmark_recognition.py`
  - [ ] Measure VRAM and throughput
  - [ ] Update docstrings
  - [ ] Create PR with results

---

## Rollback Plan

If PaddleOCR integration fails:

1. **Immediate**: Remove `paddleocr` from `pyproject.toml`
2. **Fallback**: Continue using `StubRecognizer` with detection-only mode
3. **Alternative**: Pivot to TrOCR backend (requires ~2 additional days)

---

## Dependencies & Blockers

| Dependency | Status | Mitigation |
|------------|--------|------------|
| PaddlePaddle CUDA 12.x | Unverified | Test in devcontainer first |
| RTX 3090 availability | ✅ Confirmed | Use user's local GPU |
| PP-OCRv5 Korean model | ✅ Available | Built-in to PaddleOCR |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Contract test pass rate | 100% | `pytest tests/unit/test_recognizer_contract.py` |
| VRAM usage | ≤ 1GB | `nvidia-smi` |
| Batch throughput (32 crops) | ≤ 400ms | Benchmark script |
| Korean accuracy (sample set) | ≥ 90% | Manual verification |

---

## Next Steps After Completion

Upon successful completion of Phase 1:

1. Create PR with all changes
2. Merge to `main` branch
3. Proceed to [Phase 2-3 Implementation Plan](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/artifacts/implementation_plans/2025-12-25_0410_implementation_plan_phase2-pipeline-integration.md)
