---
ads_version: "1.0"
type: "bug_report"
category: "troubleshooting"
status: "completed"
severity: "critical"
version: "1.0"
tags: ['hydra', 'configuration', 'initialization', 'startup']
title: "OCRModel Crash on Hydra Composition with DictConfig"
date: "2026-01-04 17:00 (KST)"
branch: "main"
summary: "OCRModel initialization crashed with KeyError or TypeError when using Hydra composition (e.g., `model/architectures: parseq`) because the `architecture_name` resolved to a DictConfig object instead of a string, and subsequent type checks failed."
---

# Details

## Symptoms
When running `uv run python runners/train.py --config-name train_parseq`, the training failed immediately with:
```text
KeyError: "Architecture '{...}' not registered..."
```
or after partial fix:
```text
TypeError: typing.Any cannot be used with isinstance()
```

## Root Cause
1. **Hydra Composition**: Using `defaults: - model/architectures/parseq` causes the `architecture_name` field in the config to convey the *entire configuration dictionary* of the architecture, not just its name string, due to how `overrides` or `@package` directives are processed in some Hydra setups.
2. **Type Handling**: The `OCRModel` code expected `architecture_name` to occupy a string or `None`, or relied on `isinstance(x, (dict, Any))` which is invalid in Python 3.11+.

## Fix Implementation
Updated `ocr/models/architecture.py`:
1. Use `ocr.utils.config_utils.is_config(obj)` to safely check for configuration objects.
2. If `architecture_name` is a config object, cache it as `self.architecture_config_obj` and extract the actual name string from keys `architecture_name` or `name`.
3. Use the cached `architecture_config_obj` as a primary source for component configuration, resolving parameters correctly.

## Verification
Confirmed via `reproduce_issue.py` script and `fast_dev_run` that `OCRModel` now correctly initializes without crashing on the name resolution.
