# Initial Analysis: Hydra Configuration and Legacy Import Issues

**Date**: 2026-01-22
**Session**: Hydra configs and legacy imports debugging
**Primary Domain**: Detection (det_resnet50_v1)
**Secondary Domain**: Recognition (rec_baseline_v1)

---

## Executive Summary

Debugging session revealed **two critical issues** blocking training pipeline execution:

1. **"Ghost Code" Phenomenon**: Runtime environment detached from source code (non-editable install)
2. **Hydra Recursive Instantiation Trap**: `_recursive_=True` causing premature optimizer instantiation

**Impact**:
- 53 broken Python imports
- 13-18 broken Hydra targets (fluctuating during fixes)
- Training pipeline completely blocked

---

## Initial Symptoms

### Detection Pipeline Failures

```bash
python runners/train.py experiment=det_resnet50_v1 +trainer.fast_dev_run=True
```

**Error Sequence**:
1. `ModuleNotFoundError: No module named 'ocr.domains.detection.utils.logging'`
2. `ModuleNotFoundError: No module named 'ocr.domains.detection.metrics.utils'`
3. `ModuleNotFoundError: No module named 'ocr.core.metrics'`

**Pattern**: Cascading import failures in detection domain modules

### Recognition Pipeline Failures

```bash
python runners/train.py experiment=rec_baseline_v1 +trainer.fast_dev_run=True
```

**Error**:
```
Error in call to target 'torch.optim.adam.Adam':
TypeError("Adam.__init__() missing 1 required positional argument: 'params'")
full_key: model.architectures.cfg.optimizer
```

**Pattern**: Hydra attempting to instantiate optimizer before model creation

---

## Audit Results

### Master Audit Scan

**Scope**:
- 390 Python files scanned
- 45 YAML configuration files scanned

### Broken Imports (53 total)

**Critical Training Pipeline Imports**:
- `ocr.domains.detection.module.DetectionPLModule` (line 101 in orchestrator.py)
- `ocr.domains.detection.evaluation.CLEvalEvaluator`
- `ocr.core.metrics.CLEvalMetric`
- `ocr.core.lightning.get_pl_modules_by_cfg`

**Utility Imports**:
- `ocr.core.utils.text_rendering.put_text_utf8`
- `ocr.core.utils.perspective_correction.*`
- `ocr.core.utils.polygon_utils.filter_degenerate_polygons`

**Inference Pipeline Imports**:
- `ocr.core.inference.orchestrator.InferenceOrchestrator`
- `ocr.core.inference.engine.InferenceEngine`
- `ocr.core.inference.image_loader.ImageLoader`

**External Dependencies** (not installed):
- `tiktoken` (3 occurrences)
- `typer` (1 occurrence)
- `deep_translator` (1 occurrence)

### Broken Hydra Targets (13-18 total)

**Primary Config**: `configs/data/datasets/craft.yaml`

**Interpolation Issues**:
- `${data.dataset_path}.ValidatedOCRDataset` → Module not found
- `${data.dataset_config_path}.DatasetConfig` → Module not found
- `${dataset_path}.DBTransforms` → Module not found
- `${dataset_path}.CraftCollateFN` → Module not found

**Missing Modules**:
- `ocr.data.datasets.preprocessing.external.create_background_removal_transform`

---

## Environment State

### Package Installation Mode

**Expected**: Editable install (`pip install -e .`)
**Actual**: Standard install (site-packages)

**Evidence**:
- Code changes not reflected in runtime
- Added print statements not appearing
- Renamed files not causing import errors
- Stack traces showing non-existent code lines

### Python Environment

- Python version: 3.x (with PyTorch 2.6.0)
- Package manager: pip (should migrate to uv)
- Installation location: `site-packages` (incorrect)

---

## Critical Observations

### 1. Physical vs. Logical Drift

**Physical Drift**: Code on disk vs. code in memory
- Workspace files being edited
- Runtime loading from site-packages
- No synchronization between the two

**Logical Drift**: Hydra configs vs. Python modules
- Configs referencing old module paths
- Modules moved during refactoring
- Interpolation variables not resolving

### 2. Cascading Failure Pattern

```
orchestrator.py:101 → DetectionPLModule import fails
    ↓
module.py:11 → WandbProblemLogger import fails
    ↓
wandb_loggers.py:13 → CLEvalMetric import fails
    ↓
cleval_metric.py:15 → logging utils import fails
```

**Implication**: Single broken import blocks entire detection pipeline

### 3. Hydra Behavior Conflict

**Codebase Design**: Factory pattern expecting raw configs
```python
def get_model_by_cfg(cfg):
    # Expects cfg.optimizer as config dict
    # Will instantiate optimizer AFTER model creation
```

**Hydra Default**: Recursive instantiation
```python
hydra.utils.instantiate(architectures, cfg=config)
# Hydra sees cfg.optimizer and tries to instantiate it immediately
# Fails because model.parameters() doesn't exist yet
```

---

## Initial Hypotheses

### Hypothesis 1: Ghost Code (CONFIRMED)

**Theory**: Package installed in standard mode, not editable
**Test**: Rename critical file, check if import fails
**Result**: Import still works → Confirmed ghost code

### Hypothesis 2: Recursive Instantiation (CONFIRMED)

**Theory**: Hydra instantiating optimizer too early
**Test**: Add `_recursive_=False` to instantiate call
**Result**: Error disappears → Confirmed

### Hypothesis 3: Refactoring Incomplete

**Theory**: Module reorganization left broken import paths
**Test**: Run master audit
**Result**: 53 broken imports → Confirmed incomplete refactoring

---

## Immediate Actions Taken

1. **Removed problematic configs**:
   ```bash
   rm configs/data/transforms/document_geometry.yaml
   rm configs/data/transforms/image_enhancement.yaml
   ```

2. **Ran multiple audit cycles** to track progress

3. **Identified pattern**: Import errors decreasing but Hydra target errors persisting

---

## Next Steps Required

1. **Fix environment**: Reinstall in editable mode
2. **Fix imports**: Systematic alignment of 53 broken imports
3. **Fix Hydra targets**: Resolve interpolation variables and update paths
4. **Add safeguards**: Prevent future ghost code scenarios

---

## References

- Audit logs: `logs/debug_log_relevant_to_pain_points.log`
- Pain points analysis: `artifacts/analysis_outputs/debugging_pain_points.md`
- Broken targets manifest: `artifacts/analysis_outputs/broken_targets.json`
- Master audit script: `scripts/audit/master_audit.py`

---

## Key Metrics

| Metric                   | Initial State               |
| ------------------------ | --------------------------- |
| Broken Python imports    | 53                          |
| Broken Hydra targets     | 18 → 13 (fluctuating)       |
| Files scanned (.py)      | 390                         |
| Files scanned (.yaml)    | 45                          |
| Training pipeline status | ❌ Blocked                   |
| Detection domain         | ❌ Import failures           |
| Recognition domain       | ❌ Hydra instantiation error |
