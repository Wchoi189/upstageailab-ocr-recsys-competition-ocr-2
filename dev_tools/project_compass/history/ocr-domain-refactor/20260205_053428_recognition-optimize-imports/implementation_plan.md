# Import Latency Optimization: Lazy Loading Implementation

Address severe startup timeouts (>60s) and import hangs encountered during Phase 3 verification by implementing lazy loading for heavy dependencies and auditing wildcard imports in the recognition pipeline.

## User Review Required

> [!WARNING]
> **Breaking Changes:**
> - All `torchmetrics` imports in recognition module will be deferred to `__init__` method
> - Lightning callback/logger imports will be moved to function-level scope
> - This may affect any external code that directly imports from modified modules

> [!IMPORTANT]
> **Design Decision: Import Time Profiling First**
> Before implementing fixes, we must run `python -X importtime` with tuna visualization to confirm the hypothesized bottlenecks (torchmetrics, lightning, scipy). This data-driven approach ensures we target the actual culprits.

## Proposed Changes

### Phase 1: Diagnostic Profiling

#### [NEW] [import_profile.log](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/import_profile.log)
- Run `uv run python -X importtime runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True 2> import_profile.log`
- Capture raw importtime data for baseline analysis

#### Tuna Visualization
- Install tuna: `uv pip install tuna`
- Generate interactive report: `tuna import_profile.log`
- Document top 5 slowest imports with timings

---

### Phase 2: Recognition Module Lazy Loading

#### [MODIFY] [module.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/recognition/module.py)
**Current Issue:** Line 5 imports `torchmetrics.text.CharErrorRate` at module level, triggering:
- BERT tokenizer compilation (transformers)
- C++ tokenizer bindings
- ~15-20s overhead

**Changes:**
1. Remove top-level import: `from torchmetrics.text import CharErrorRate`
2. Move to `__init__` method lazy initialization:
   ```python
   def __init__(self, model, dataset, config, metric_cfg=None):
       super().__init__(model, dataset, config, metric_cfg)

       # Lazy import - only loads when RecognitionPLModule is instantiated
       from torchmetrics.text import CharErrorRate
       self.rec_cer = CharErrorRate()
   ```

**Rationale:** Recognition module is only instantiated during actual training, not during config parsing or model inspection.

---

### Phase 3: Lightning Orchestrator Optimization

#### [MODIFY] [orchestrator.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/pipelines/orchestrator.py)
**Current Issue:** Line 12 imports `lightning.pytorch.Trainer` at module level, loading entire Lightning ecosystem (DeepSpeed, XLA strategies).

**Changes:**
1. Remove top-level import: `from lightning.pytorch import Trainer`
2. Move to `setup_trainer()` method (line 149):
   ```python
   def setup_trainer(self):
       """Build PyTorch Lightning Trainer from V5.0 Hydra configs."""
       from lightning.pytorch import Trainer  # Lazy import

       logger.info("⚡ Configuring Lightning Trainer...")
       # ... rest of method
   ```

**Impact:** This module is imported by `runners/train.py`, so deferring Trainer import eliminates strategy loading until trainer is actually needed.

---

### Phase 4: Analysis Scripts Deferred Loading

#### [MODIFY] [analyze_worst_images.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/analysis/validation/analyze_worst_images.py)
**Current Issue:** Line 9 imports `matplotlib.pyplot` at module level.

**Changes:**
1. Remove top-level import
2. Add local import inside visualization functions:
   ```python
   def plot_worst_performers(data):
       import matplotlib.pyplot as plt  # Lazy import
       # ... plotting logic
   ```

#### [MODIFY] [visualize.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/domains/detection/analysis/visualize.py)
**Same pattern:** Move matplotlib to function scope.

**Rationale:** Analysis scripts are rarely used during training runs and should not block startup.

---

### Phase 5: Verification Baseline Restoration

#### [MODIFY] [train.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train.py)
**Cleanup:** Remove debug print statements (lines 10-12):
```python
from ocr.core.lightning.base import OCRPLModule
import inspect
print(f"DEBUG: OCRPLModule loaded from: {inspect.getfile(OCRPLModule)}")
```

**Rationale:** These were added for debugging and are no longer needed.

---

## Verification Plan

### Automated Tests

#### 1. Import Time Profiling (Pre/Post Comparison)
**Command:**
```bash
# Baseline (before changes)
uv run python -X importtime runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True 2> import_baseline.log

# After optimization
uv run python -X importtime runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True 2> import_optimized.log

# Compare using tuna
tuna import_baseline.log  # Note slowest modules
tuna import_optimized.log # Verify improvements
```

**Success Criteria:**
- Total import time < 15 seconds
- No single module takes > 5 seconds
- `torchmetrics` and `lightning` imports deferred (should not appear in top 10)

#### 2. Fast Dev Run Execution
**Command:**
```bash
uv run python runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True
```

**Success Criteria:**
- Training loop starts within 15 seconds
- No import errors or missing module exceptions
- Metrics (CER, accuracy) are computed correctly
- Console output shows proper metric logging

#### 3. Syntax and Import Validation
**Command:**
```bash
# Check syntax
uv run python -m py_compile ocr/domains/recognition/module.py
uv run python -m py_compile ocr/pipelines/orchestrator.py

# Verify imports work
uv run python -c "from ocr.domains.recognition.module import RecognitionPLModule; print('✓ RecognitionPLModule imports successfully')"
uv run python -c "from ocr.pipelines.orchestrator import OCRProjectOrchestrator; print('✓ OCRProjectOrchestrator imports successfully')"
```

**Success Criteria:**
- All files compile without syntax errors
- Modules can be imported without triggering heavy dependency loading

### Manual Verification

> [!NOTE]
> **Optional Full Training Run**
> If the user wants to verify end-to-end behavior, they can run a full epoch:
> ```bash
> uv run python runners/train.py experiment=rec_baseline_v1 trainer.max_epochs=1
> ```
> This is NOT required for verifying the optimization but can validate that lazy loading doesn't break training.

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Lazy imports break pickling/serialization | Lightning modules are not pickled during fast_dev_run; full checkpoint verification in Phase 3 |
| Third-party code expects eager imports | All modified modules are internal; no public API contracts broken |
| IDE auto-completion degradation | Use `TYPE_CHECKING` blocks to preserve type hints |

---

## Rollback Plan

If verification fails, revert changes to:
- `ocr/domains/recognition/module.py` (restore line 5)
- `ocr/pipelines/orchestrator.py` (restore line 12)
- Analysis scripts (restore top-level matplotlib imports)

Git commit message: `"Revert: Import optimization (failed verification)"`
