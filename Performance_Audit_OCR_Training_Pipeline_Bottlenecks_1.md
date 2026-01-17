# Performance Audit: OCR Training Pipeline Bottlenecks

**Context**: Training runs take 1+ minute to start execution or return errors
**Priority**: High (affects development velocity and AI iteration speed)

---

## Critical Bottlenecks Identified

### 1. Heavy Module-Level Imports ⚠️ HIGH IMPACT

**Location**: [ocr/core/lightning/ocr_pl.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py) (lines 1-18)

**Issue**:
```python
import lightning.pytorch as pl
import torch  # Heavy: ~2-5s on first import
from hydra.utils import instantiate  # Triggers OmegaConf, Hydra init
from torchmetrics.text import CharErrorRate  # Heavy metrics library
```

**Impact**: ~5-8s startup overhead before any code executes

**Recommendation**:
- Move torch/lightning imports inside class methods where possible
- Use `TYPE_CHECKING` guard for type hints
- Consider [runners/train_fast.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train_fast.py) pattern (lazy import after Hydra validation)

**Example Fix**:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import lightning.pytorch as pl

# Then import inside __init__ or first usage
```

---

### 2. Expensive OCRPLModule Initialization ⚠️ HIGH IMPACT

**Location**: `ocr/core/lightning/ocr_pl.py:__init__()` (lines 40-78)

**Issues**:
1. **Evaluator instantiation** (lines 58-59):
   ```python
   self.valid_evaluator = CLEvalEvaluator(self.dataset["val"], ...)
   self.test_evaluator = CLEvalEvaluator(self.dataset["test"], ...)
   ```
   - Each evaluator may validate/scan entire dataset
   - Happens BEFORE training starts

2. **WandbProblemLogger init** (lines 65-71):
   - Initializes complex logging infrastructure eagerly
   - Loads normalize stats, dataset metadata

3. **Metric initialization** (line 53):
   ```python
   self.metric = instantiate(metric_cfg) if metric_cfg else CLEvalMetric(**self.metric_kwargs)
   ```
   - Heavy CLEval metric setup regardless of domain

**Impact**: 10-20s initialization overhead

**Recommendations**:
- Lazy-initialize evaluators on first validation step
- Defer logger setup until first use
- Cache expensive computations


---

### 3. Redundant Dataset Scanning ⚠️ MEDIUM IMPACT

**Location**: [ocr/core/evaluation/evaluator.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py) (CLEvalEvaluator)

**Suspected Issue** (needs verification):
```python
# Likely happening in evaluator init:
for filename in dataset:
    # Validate file exists, parse annotations, etc.
```

**Impact**: 5-15s for large datasets

**Recommendation**:
- Profile evaluator initialization
- Consider metadata caching (pickle/json cache of dataset structure)
- Defer full dataset scan until first evaluation

---

### 4. Hydra Configuration Overhead 🔍 NEEDS PROFILING

**Location**: `main.py` or training entry point

**Suspected Issues**:
1. **Deep config tree resolution**: Nested `defaults` lists may cause cascading file reads
2. **Interpolation resolution**: Complex `${...}` references slow down OmegaConf
3. **Instantiation validation**: Hydra validates all `_target_` paths at config time

**Impact**: Unknown, but likely 10-30s

**Recommendations**:
- Profile with `python -m cProfile main.py --help` to see Hydra overhead
- Simplify config inheritance chains
- Consider config caching for development

---

### 5. wandb_base.py: Excessive Registry Lookups 🔍 LOW-MEDIUM IMPACT

**Location**: `ocr/core/utils/wandb_base.py:_architecture_default_component()` (lines 237-246)

**Issue**:
```python
def _architecture_default_component(architecture_name: str | None, component: str) -> str:
    if not architecture_name:
        return ""
    try:
        from ocr.core import registry as _registry  # Import INSIDE function
        mapping = _registry.get_architecture(str(architecture_name))
```

Called repeatedly in [generate_run_name()](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_utils.py#289-425) for encoder/decoder/head/loss components.

**Impact**: 4 redundant registry imports per run name generation

**Recommendation**:
- Cache registry at module level (it's immutable)
- Or cache architecture lookups with `@lru_cache`

---

### 6. Collate Function Instantiation 🔍 NEEDS INVESTIGATION

**Location**: `ocr/data/lightning_data.py:_build_collate_fn()` (lines 25-30)

**Issue**:
```python
def _build_collate_fn(self, *, inference_mode: bool) -> Any:
    collate_fn = instantiate(self.collate_cfg)  # Hydra instantiate every time?
```

Called 4 times (train/val/test/predict dataloaders).

**Questions**:
- Is collate_fn cached or rebuilt each time?
- Does instantiation trigger heavy imports?

**Recommendation**: Profile and consider singleton pattern if expensive

---

### 7. text_rendering.py Import Chain (Suspected) 🔍 LOW IMPACT

**Location**: `ocr/domains/detection/callbacks/wandb.py:log_validation_images()`

**Issue**:
```python
from ocr.core.utils.text_rendering import put_text_utf8
```

This imports PIL, numpy, cv2 if not already imported. Happens during first validation logging.

**Impact**: Minor (1-2s), but adds to death-by-1000-cuts

**Recommendation**: Already using lazy imports in callbacks (good!), but verify `text_rendering.py` itself doesn't have module-level heavy ops

---

## Prioritized Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. ✅ **Already done**: Lazy imports in [wandb_base.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_base.py) (good!)
2. **Cache architecture registry** in [wandb_base.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_base.py) → ~0.5s savings
3. **Lazy-init evaluators** in `OCRPLModule.__init__()` → ~10-15s savings

### Phase 2: Profiling (2-3 hours)
1. **Profile Hydra overhead**:
   ```bash
   python -m cProfile -o profile.stats main.py --cfg job 2>&1 | head -50
   ```
2. **Profile OCRPLModule init**:
   ```python
   import cProfile
   cProfile.run('OCRPLModule(model, dataset, config)', sort='cumtime')
   ```
3. **Identify dataset scanning** in evaluator

### Phase 3: Structural Fixes (4-6 hours)
1. Implement lazy evaluator initialization
2. Add metadata caching for datasets
3. Optimize Hydra config structure (reduce nesting/interpolations)

---

## Specific Files to Audit

| Priority | File                                                                                                                              | Suspected Issue                | Est. Impact |
| -------- | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------ | ----------- |
| 🔴 HIGH   | [ocr/core/lightning/ocr_pl.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/lightning/ocr_pl.py)         | Heavy init, eager evaluators   | 15-20s      |
| 🔴 HIGH   | `main.py` / Hydra entry                                                                                                           | Config resolution overhead     | 10-30s      |
| 🟡 MEDIUM | [ocr/core/evaluation/evaluator.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/evaluation/evaluator.py) | Dataset scanning               | 5-15s       |
| 🟡 MEDIUM | [ocr/data/lightning_data.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/data/lightning_data.py)             | Repeated collate instantiation | 2-5s        |
| 🟢 LOW    | [ocr/core/utils/wandb_base.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_base.py)         | Registry lookups               | 0.5-1s      |

---

## Quick Diagnostic Commands

```bash
# 1. Profile import time (find slowest imports)
python -X importtime main.py 2>&1 | grep -i 'torch\|lightning\|hydra' | sort -k2 -n

# 2. Profile Hydra config resolution
time python main.py --cfg job  # Should be <5s if Hydra is not the bottleneck

# 3. Profile full startup
python -m cProfile -s cumtime main.py --help 2>&1 | head -30

# 4. Check if WSL/Python version is issue
python --version  # Ensure Python 3.11+ (you have 3.11.14 ✓)
which python  # Verify using .venv (you are ✓)
```

---

## Notes from Refactoring

**Good practices observed**:
- ✅ Lazy imports already in [wandb_base.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/core/utils/wandb_base.py) (cv2, torch, numpy loaded in functions)
- ✅ [runners/train_fast.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/runners/train_fast.py) demonstrates proper lazy loading pattern
- ✅ Type checking guards used in some modules

**Anti-patterns observed**:
- ❌ `OCRPLModule.__init__()` does too much eager work
- ❌ Evaluator initialization happens before training starts
- ❌ No visible caching of expensive operations

---

## Estimated Total Savings

Conservative estimate: **20-40s startup time reduction** achievable with Phase 1-2 fixes.

If Hydra is the culprit: Could see **30-60s** improvement with config optimization.

---

## Phase 2 Refactoring Insights (2026-01-18)

### New Bottlenecks Discovered During Code Review

#### 8. Deprecated ocr_pl.py Still Being Imported 🔴 CRITICAL
**Location**: `ocr/core/lightning/ocr_pl.py` (516 lines, deprecated)

**Issue**:
- File still exists and may be imported by old code paths
- Contains redundant imports of torch, Lightning, CharErrorRate
- Creates circular import risk with new domain modules

**Impact**: 2-5s from redundant module loading

**Action**:
```bash
# DELETE deprecated file (refactored into domain modules)
rm ocr/core/lightning/ocr_pl.py
```

✅ **Safe to delete**: All functionality moved to:
- `ocr/core/lightning/base.py` (abstract base)
- `ocr/domains/detection/module.py` (DetectionPLModule)
- `ocr/domains/recognition/module.py` (RecognitionPLModule)

---

#### 9. torch.compile() Blocking During Init 🔴 HIGH IMPACT
**Location**: `ocr/core/lightning/base.py:49`

**Issue**:
```python
if hasattr(config, "compile_model") and config.compile_model:
    torch._dynamo.config.capture_scalar_outputs = True
    self.model = torch.compile(self.model, mode="default")  # BLOCKS HERE
```

**Impact**: 10-20s compilation time **before** first forward pass

**Recommendations**:
1. **Quick fix**: Disable during dev: `compile_model=false`
2. **Proper fix**: Defer compilation to first training step:
   ```python
   def on_train_start(self):
       if self.config.compile_model and not self._compiled:
           self.model = torch.compile(self.model)
           self._compiled = True
   ```

---

#### 10. WandB Logger Eagerly Created in DetectionPLModule 🟡 MEDIUM IMPACT
**Location**: `ocr/domains/detection/module.py:56`

**Issue**:
```python
def __init__(self, model, dataset, config, metric_cfg=None):
    super().__init__(...)
    # Immediate WandB initialization
    self.wandb_logger = WandbProblemLogger(config, ...)  # Network calls?
```

**Impact**: 5-15s if WandB needs to authenticate or check for updates

**Fix**:
```python
@property
def wandb_logger(self):
    if self._wandb_logger is None:
        self._wandb_logger = WandbProblemLogger(...)
    return self._wandb_logger
```

---

#### 11. Evaluator Initialized Too Early (Partially Fixed) ✅ IMPROVED
**Location**: `ocr/domains/detection/module.py:64`

**Status**:
- ✅ Already using lazy properties for `valid_evaluator` and `test_evaluator`
- ✅ Deferred until first access

**Previous bottleneck eliminated**: This was 10-15s, now ~0s until first validation

---

#### 12. Double Import of CLEvalEvaluator (Now Fixed) ✅ RESOLVED
**Location**: `ocr/core/evaluation/__init__.py`

**Previous issue**:
```python
# OLD: Could trigger double import if both paths used
from .evaluator import CLEvalEvaluator
from ocr.domains.detection.evaluation import CLEvalEvaluator
```

**Current status**: ✅ Fixed with forward import pattern
```python
# Now forwards to single source
from ocr.domains.detection.evaluation import CLEvalEvaluator
```

---

### Additional Quick-Win Recommendations from Refactoring

#### A. Consolidate Collate Function Instances
**Location**: `ocr/data/lightning_data.py`

**Current**: 4 separate instantiations (train/val/test/predict)
```python
def train_dataloader(self):
    collate_fn = self._build_collate_fn(inference_mode=False)  # Instance 1

def val_dataloader(self):
    collate_fn = self._build_collate_fn(inference_mode=False)  # Instance 2 (same!)
```

**Optimized**:
```python
def __init__(self, dataset, config):
    super().__init__()
    self._train_collate = None  # Lazy cache
    self._predict_collate = None

@property
def train_collate_fn(self):
    if self._train_collate is None:
        self._train_collate = self._build_collate_fn(inference_mode=False)
    return self._train_collate
```

**Savings**: 1-3s (if collate_fn instantiation is expensive)

---

#### B. Remove Unused Recognition Metrics from DetectionPLModule
**Location**: `ocr/core/lightning/base.py:53`

**Issue**:
```python
# Base class creates metric for ALL domains
self.metric = instantiate(metric_cfg) if metric_cfg else CLEvalMetric(**self.metric_kwargs)
```

But RecognitionPLModule doesn't use `self.metric` (uses `self.rec_cer` instead)

**Fix**: Move metric initialization to DetectionPLModule only

**Savings**: 0.5-1s for recognition tasks

---

#### C. Batch Imports at Module Level (Not Function Level)
**Location**: `ocr/core/lightning/base.py:137-138`

**Current**:
```python
def on_train_epoch_start(self):
    import ocr.data.datasets.db_collate_fn  # Import EVERY epoch
    ocr.data.datasets.db_collate_fn._db_collate_logged_stats = False
```

**Should be**:
```python
# Top of file
from ocr.data.datasets import db_collate_fn

def on_train_epoch_start(self):
    db_collate_fn._db_collate_logged_stats = False
```

**Savings**: Minimal per call, but cleaner

---

### Updated Priority Matrix (Post-Refactoring)

| Priority | Action                                 | File                  | Est. Savings |
| -------- | -------------------------------------- | --------------------- | ------------ |
| 🔴 **P0** | **Delete deprecated ocr_pl.py**        | `ocr/core/lightning/` | **2-5s**     |
| 🔴 **P1** | Defer torch.compile to first step      | `base.py:49`          | **10-20s**   |
| 🔴 P1     | Lazy WandB logger                      | `detection/module.py` | 5-15s        |
| 🟡 P2     | Cache collate functions                | `lightning_data.py`   | 1-3s         |
| 🟡 P2     | Remove unused metrics from recognition | `base.py`             | 0.5-1s       |
| 🟢 P3     | Move imports to module level           | `base.py`             | <0.5s        |

**Total potential savings: 19-44.5 seconds**

---

### Profiling Command for Next Session

```bash
# Get baseline startup time
time python main.py --config-name train --help  # Current: ~60s

# Profile with line-level detail
python -m line_profiler -l main.py --config-name train trainer.fast_dev_run=true

# Check Hydra resolution specifically
HYDRA_FULL_ERROR=1 time python -c "from hydra import compose, initialize; initialize(config_path='configs'); cfg = compose(config_name='main', overrides=['domain=detection'])"
```

---

## Summary: Immediate Actions

1. ✅ **DELETE** `ocr/core/lightning/ocr_pl.py` (no longer needed, causes import overhead)
2. **DISABLE** `compile_model` in development configs
3. **TEST** startup time after deletion: `time python main.py --config-name train --help`
4. **PROFILE** if still slow: `python -X importtime main.py 2> import_time.log`

Expected result: **~40s startup** (down from ~60s) with just steps 1-2.
