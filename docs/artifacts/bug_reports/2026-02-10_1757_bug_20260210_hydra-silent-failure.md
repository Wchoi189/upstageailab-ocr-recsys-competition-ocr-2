---
ads_version: 1.0
type: bug_report
category: troubleshooting
status: completed
version: 1.0
severity: critical
description: Critical silent failure in Hydra Config composition where flat logger configs are mis-instantiated as list of strings, causing no logging without error.
tags: critical,hydra,config,observability
title: Hydra Logger Config Silent Failure
date: 2026-02-10 17:57 (KST)
branch: main
---

# Bug Report - Hydra Logger Config Silent Failure
Bug ID: BUG-20260210-SILENT-HYDRA

## Summary
Training runs execute without logging metrics, artifacts, or images, leading to wasted compute and loss of experimental data. This is caused by a Hydra configuration composition mismatch where a flat `DictConfig` is provided when a `List` or `Dict` of configs is expected by the Orchestrator.

## Environment
- **OS/Env**: Linux / Docker
- **Dependencies**: Hydra 1.3+, PyTorch Lightning

## Reproduction
Command:
```bash
uv run python scripts/runners/train.py ... +train/logger=wandb
```
(Where `configs/train/logger/wandb.yaml` lacks the `@package` wrapper).

## Comparison
**Expected**: `self.cfg.train.logger` is a Dictionary of logger configurations (e.g., `{wandb: {...}, tensorboard: {...}}`).
**Actual**: `+train/logger=wandb` loads a *single* config object at `self.cfg.train.logger`.

## Mechanism of Silent Failure
Iterating over `self.cfg.train.logger.values()` iterates over the *properties* of the single WandB config (strings, ints). `hydra.utils.instantiate()` on these primitives returns them as-is (or creates garbage objects) without raising an exception. The `Trainer` receives a list of strings/garbage instead of `Logger` instances.

## Fix Implemented
Created `configs/train/logger/single_wandb.yaml` which wraps the config:
```yaml
# @package _group_
defaults:
  - wandb@_group_.wandb
```

## 🛑 Prevention Tool Design: "Silent Failure Detector"

To prevent this class of errors (Structural Type Mismatch) in the future, we propose a validation phase.

### Tool Concept: `scripts/utils/validate_config.py`

**Goal:** Detect structure/type mismatches *before* instantiation.

**Logic:**
1.  **Load Config:** Resolve the full Hydra config without instantiating.
2.  **Schema Validation:** Check specific keys against expected types.
    - `train.logger`: Must be `DictConfig` or `ListConfig` **containing** configs (check for keys like `_target_` *inside* the values, not at the top level).
    - `train.callbacks`: Similar check.
3.  **Fail Fast:** If `train.logger` has `_target_` at its root, it is **WRONG** (it should be a container *of* targets).

**Implementation Snippet:**
```python
def validate_structure(cfg):
    # Check Logger Structure
    if cfg.train.get("logger"):
        logging_conf = cfg.train.logger
        # If the container ITSELF has a target, it's likely a flat config (Bug!)
        if "_target_" in logging_conf:
             raise RuntimeError(
                 "CRITICAL CONFIG ERROR: 'train.logger' seems to be a single Logger config. "
                 "It MUST be a dictionary/list of loggers. "
                 "Did you forget a nesting wrapper?"
             )
```

**Integration:**
Call this validator at the start of `OCRProjectOrchestrator.__init__`.
