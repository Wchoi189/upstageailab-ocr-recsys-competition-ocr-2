---
ads_version: "1.0"
title: "001 Startup Latency SLA Violation"
date: "2025-12-23 09:00 (KST)"
type: bug_report
category: troubleshooting
status: completed
version: "1.0"
tags: ['bug_report', 'performance', 'latency']
---

# BUG-20251223-001: 3-Minute Startup Delay due to Eager PyTorch Import

## Overview
**Date Detected:** 2025-12-23
**Severity:** Critical (Violates 10s Startup SLA)
**Status:** Resolved

## Description
The OCR Inference Console backend exhibited excessive startup times ranging from 14 seconds (local environment) to 3 minutes (user report). Profiling revealed that the `ocr.inference` package and its dependencies were eagerly importing `torch`, `torchvision`, and other heavy libraries at the module level. This meant that simply importing `apps.ocr_inference_console.backend.main` triggered a full PyTorch initialization before the FastAPI application could even start.

## Impact
- Backend startup time exceeded the 10-second SLA.
- Delayed availability of health check endpoints.
- Poor developer experience during iteration.

## Root Cause
The `ocr/inference/dependencies.py` module performed top-level imports of `torch` and `lightning.pytorch`. This module was imported by `model_manager.py`, which was imported by `orchestrator.py`, which was imported by `inference_service.py`, creating a chain of hard dependencies that blocked startup.

## Resolution
Refactored the inference architecture to implement **Lazy Loading**:
1.  **`ocr/inference/dependencies.py`**: Removed eager imports. Implemented `importlib.util.find_spec` to check for module availability without importing.
2.  **`ocr/inference/model_manager.py`**: Moved `import torch` inside methods that require it (e.g., `load_model`, `cleanup`).
3.  **`ocr/inference/preprocess.py` & `postprocess.py`**: Moved usage of `torch` and `transforms` inside functional scopes.

## Verification
- **Profiler Results**: `scripts/performance/profile_checks.py` shows `apps.ocr_inference_console.backend.main` now imports in **~0.17s** (down from >14s) without triggering `torch` load.
- **Benchmarks**: Verified that inference still functions correctly despite the deferred imports.

## Future Work
- Implement background model warming to mitigate the "cold start" latency (now shifted to the first inference request).
