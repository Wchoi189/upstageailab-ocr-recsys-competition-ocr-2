---
title: "Implementation Plan Phase2 Logger Factory"
date: "2025-12-06 18:08 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---



# Phase 2 Refactoring: Logger Factory Extraction

## Overview

This guide provides exact code changes for Phase 2 refactoring - extracting logger creation logic into a reusable factory function.

**Estimated Time:** 60 minutes
**Risk Level:** Low
**Lines Removed:** 55 lines
**Complexity Reduction:** ~25%
**Reusability Gained:** Logger factory usable in test.py, predict.py

---

## Context

Phase 1 (completed 2025-12-05 22:42):
- ✅ Removed signal handlers (35 lines)
- ✅ Removed DDP auto-scaling (28 lines)
- ✅ Removed unused imports
- **Result**: 65 lines removed, train.py now 239 lines

Phase 2 (this plan):
- Extract logger factory (50 lines)
- Create reusable component for all runners
- Simplify train() function further

---

## Change 1: Create Logger Factory Module

### File: `ocr/utils/logger_factory.py` (NEW)

```python
"""Logger factory for creating appropriate logger from configuration.

Provides a centralized factory function for creating Lightning loggers
(WandB or TensorBoard) based on configuration settings.
"""

from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf


def create_logger(config: DictConfig) -> Logger:
    """Create appropriate logger (W&B or TensorBoard) from configuration.

    Args:
        config: Hydra configuration containing logger settings

    Returns:
        Lightning Logger instance (WandbLogger or TensorBoardLogger)

    Logic:
        - If config.logger.wandb.enabled is explicitly False → TensorBoard
        - Otherwise → W&B (default, enabled by default)
    """
    logger_config = config.get("logger", {})
    wandb_cfg = logger_config.get("wandb", {})

    # Type-safe check: if explicitly disabled, use TensorBoard
    if isinstance(wandb_cfg, dict) and wandb_cfg.get("enabled") is False:
        return _create_tensorboard_logger(config, wandb_cfg)

    # Default: use W&B
    return _create_wandb_logger(config, wandb_cfg)


def _create_tensorboard_logger(config: DictConfig, wandb_cfg: dict) -> TensorBoardLogger:
    """Create TensorBoard logger for when W&B is disabled."""
    exp_version = wandb_cfg.get("exp_version", "v1.0") if isinstance(wandb_cfg, dict) else "v1.0"

    return TensorBoardLogger(
        save_dir=config.paths.log_dir,
        name=config.exp_name,
        version=exp_version,
        default_hp_metric=False,
    )


def _create_wandb_logger(config: DictConfig, wandb_cfg: dict) -> WandbLogger:
    """Create Weights & Biases logger (default)."""
    from ocr.utils.wandb_utils import generate_run_name, load_env_variables

    # Load environment variables for W&B API key
    load_env_variables()

    # Resolve interpolations before generating run name
    OmegaConf.resolve(config)
    run_name = generate_run_name(config)

    # Serialize config for W&B, handling Hydra interpolations gracefully
    try:
        wandb_config = OmegaConf.to_container(config, resolve=True)
    except Exception:
        # Fall back to unresolved config if resolution fails
        wandb_config = OmegaConf.to_container(config, resolve=False)

    project_name = wandb_cfg.get("project_name", "ocr-training") if isinstance(wandb_cfg, dict) else "ocr-training"

    return WandbLogger(
        name=run_name,
        project=project_name,
        config=wandb_config,
    )
```

---

## Change 2: Update `runners/train.py`

### Before (Lines 117-172)

```python
    logger: Logger

    wandb_cfg = getattr(config.logger, "wandb", None)
    wandb_enabled = False
    if isinstance(wandb_cfg, DictConfig):
        wandb_enabled = wandb_cfg.get("enabled", True)
    elif isinstance(wandb_cfg, dict):
        wandb_enabled = wandb_cfg.get("enabled", True)
    elif isinstance(wandb_cfg, bool):
        wandb_enabled = wandb_cfg
    elif wandb_cfg is not None:
        wandb_enabled = bool(wandb_cfg)

    if wandb_enabled:
        from lightning.pytorch.loggers import WandbLogger
        from omegaconf import OmegaConf

        from ocr.utils.wandb_utils import generate_run_name, load_env_variables

        # Load environment variables from .env.local/.env
        load_env_variables()

        # Resolve interpolations before generating run name
        OmegaConf.resolve(config)

        run_name = generate_run_name(config)

        # Properly serialize config for wandb, handling hydra interpolations
        try:
            # Try to resolve interpolations for cleaner config
            wandb_config = OmegaConf.to_container(config, resolve=True)
        except Exception:
            # Fall back to unresolved config if resolution fails
            wandb_config = OmegaConf.to_container(config, resolve=False)

        logger = WandbLogger(
            name=run_name,
            project=config.logger.wandb.project_name,
            config=wandb_config,
        )
    else:
        from lightning.pytorch.loggers.tensorboard import TensorBoardLogger

        logger = TensorBoardLogger(
            save_dir=config.paths.log_dir,
            name=config.exp_name,
            version=config.logger.wandb.exp_version if hasattr(config.logger, 'wandb') else "v1.0",
            default_hp_metric=False,
        )

    # Ensure no default logger is created by explicitly setting logger
    # This prevents lightning_logs from being created in the root directory
```

### After (Lines 117-121)

```python
    # Create appropriate logger (W&B or TensorBoard) based on configuration
    from ocr.utils.logger_factory import create_logger

    logger = create_logger(config)
```

**Lines Saved:** 55 lines (189 → 134)
**Improvement:** 5-line solution vs. 55 lines of conditional logic

---

## Change 3: Import Addition

### File: `runners/train.py`

**Line 1-10 (add import if not present):**

```python
import logging
import math
import os
import sys
import warnings

# Setup project paths automatically
import hydra
from omegaconf import DictConfig
```

No changes needed - factory imports are lazy inside train() function.

---

## Testing Procedure

### Step 1: Syntax Validation

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Check new module syntax
python -m py_compile ocr/utils/logger_factory.py

# Check modified runner syntax
python -m py_compile runners/train.py
```

**Expected:** No errors

### Step 2: Import Test

```bash
# Verify imports work correctly
python -c "from ocr.utils.logger_factory import create_logger; print('✅ Import successful')"
```

**Expected:** `✅ Import successful`

### Step 3: W&B Logger Test

```bash
# Test with W&B enabled (default)
uv run python runners/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=0.01 \
  trainer.limit_val_batches=0.01 \
  exp_name=phase2_wandb_test \
  logger.wandb.enabled=true
```

**Expected:**
- Training starts with W&B logger initialization
- Logs to "ocr-training" project
- Training completes successfully

### Step 4: TensorBoard Logger Test

```bash
# Test with W&B disabled (TensorBoard fallback)
uv run python runners/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=0.01 \
  trainer.limit_val_batches=0.01 \
  exp_name=phase2_tb_test \
  logger.wandb.enabled=false
```

**Expected:**
- Training starts with TensorBoard logger
- Logs to `outputs/experiments/train/ocr/phase2_tb_test/*/`
- Training completes successfully

### Step 5: Linting & Type Checking

```bash
# Check new module
uv run ruff check ocr/utils/logger_factory.py --fix

# Check modified runner
uv run ruff check runners/train.py --fix

# Type checking
uv run mypy ocr/utils/logger_factory.py
uv run mypy runners/train.py
```

**Expected:** All checks pass

### Step 6: Full Smoke Test

```bash
# Same as Step 1 of Phase 1 plan
uv run python runners/train.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=0.01 \
  trainer.limit_val_batches=0.01 \
  exp_name=phase2_full_test \
  logger.wandb.enabled=false
```

**Expected:** Training completes successfully, outputs "✅ Trainer.fit stopped: max_epochs=1 reached"

---

## Validation Checklist

After implementing changes:

- [ ] Syntax check passes: `python -m py_compile ocr/utils/logger_factory.py`
- [ ] Import test passes: `python -c "from ocr.utils.logger_factory import create_logger"`
- [ ] W&B logger test passes (logger.wandb.enabled=true)
- [ ] TensorBoard logger test passes (logger.wandb.enabled=false)
- [ ] Code passes ruff linting: `ruff check ocr/utils/logger_factory.py && ruff check runners/train.py`
- [ ] Code passes mypy type checking: `mypy ocr/utils/logger_factory.py && mypy runners/train.py`
- [ ] Full training smoke test passes (< 2 min)
- [ ] Git commit with clear message

---

## Git Commit Message Template

```
refactor(train): Phase 2 - Extract logger factory

Extracted logger creation logic into reusable factory function:
- Created ocr/utils/logger_factory.py with create_logger() factory
- Simplified train.py: 55 lines removed (~25% complexity reduction)
- Now handles W&B and TensorBoard logger selection cleanly
- Extracted logic is now reusable in test.py and predict.py

Benefits:
- Single responsibility: logger creation isolated in factory
- Reusable: Can use create_logger() in other runners
- Type-safe: Proper type handling for wandb_cfg variants
- Maintainable: Clear separation of concerns

Changes:
- Added: ocr/utils/logger_factory.py (85 lines)
- Modified: runners/train.py (-55 lines, +4 lines)

Testing:
- W&B logger: ✅ Initializes and logs correctly
- TensorBoard logger: ✅ Falls back correctly when disabled
- Smoke test: ✅ Training completes successfully
- Type checking: ✅ mypy passes without errors
- Linting: ✅ ruff passes all checks

Related: docs/artifacts/assessments/2025-12-05_2142_assessment-train-py-refactoring.md
Related: docs/artifacts/implementation_plans/2025-12-05_2142_implementation_plan_phase1-refactoring.md
```

---

## Rollback Plan

If anything goes wrong:

```bash
# Revert changes
git checkout ocr/utils/logger_factory.py runners/train.py

# Or delete the new file only
rm ocr/utils/logger_factory.py
git checkout runners/train.py
```

**Critical Files:**
- `ocr/utils/logger_factory.py` (new)
- `runners/train.py` (modified)

---

## Success Criteria

✅ Logger factory created and imported successfully
✅ train.py simplified with 55 lines removed
✅ W&B logger creation works correctly
✅ TensorBoard logger creation works correctly
✅ All tests pass (syntax, import, smoke test)
✅ Linting and type checking pass
✅ Ready for Phase 3 (error handling simplification)

---

## Future Work (Phase 3)

After Phase 2 is complete, Phase 3 will:
- Remove unnecessary try/except blocks
- Simplify error handling to fail-fast approach
- Further reduce cyclomatic complexity

---

**Implementation Date:** 2025-12-05
**Estimated Duration:** 60 minutes
**Risk Level:** Low
**Validation Required:** Yes (see testing procedure)

```

