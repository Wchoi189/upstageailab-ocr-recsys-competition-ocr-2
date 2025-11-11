#!/usr/bin/env python3
"""Test individual imports to identify which one causes the hang."""

import sys
import time

def test_import(module_name, import_statement):
    """Test a single import and report timing."""
    print(f"\n[TEST] Testing import: {module_name}")
    print(f"[TEST] Statement: {import_statement}")
    sys.stdout.flush()

    start_time = time.time()
    try:
        exec(import_statement)
        elapsed = time.time() - start_time
        print(f"[OK] Import successful in {elapsed:.2f}s")
        sys.stdout.flush()
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] Import failed after {elapsed:.2f}s: {e}")
        sys.stdout.flush()
        return False

# Test imports in order
imports_to_test = [
    ("logging", "import logging"),
    ("math", "import math"),
    ("os", "import os"),
    ("signal", "import signal"),
    ("sys", "import sys"),
    ("warnings", "import warnings"),
    ("hydra", "import hydra"),
    ("lightning.pytorch", "import lightning.pytorch as pl"),
    ("lightning.pytorch.callbacks", "from lightning.pytorch.callbacks import LearningRateMonitor"),
    ("omegaconf", "from omegaconf import DictConfig"),
    ("wandb", "import wandb"),
    ("ocr.utils.path_utils", "from ocr.utils.path_utils import get_path_resolver, setup_project_paths"),
    ("ocr.lightning_modules", "from ocr.lightning_modules import get_pl_modules_by_cfg"),
]

print("[DEBUG] Starting import tests...")
print("[DEBUG] This will help identify which import causes the hang")
sys.stdout.flush()

for module_name, import_stmt in imports_to_test:
    success = test_import(module_name, import_stmt)
    if not success:
        print(f"\n[STOPPED] Failed at: {module_name}")
        sys.exit(1)

    # Add a small delay to see progress
    time.sleep(0.1)

print("\n[SUCCESS] All imports completed successfully!")
sys.stdout.flush()

