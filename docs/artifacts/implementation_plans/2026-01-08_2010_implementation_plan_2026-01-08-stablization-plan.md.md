---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: "stabilization, refactor, fix"
title: "Post-Refactor Stabilization Plan"
date: "2026-01-08 20:10 (KST)"
branch: "main"
description: "Plan to fix the immediate test failures and stabilize the codebase post-refactor."
---

# Post-Refactor Stabilization Plan

## Goal Description
The goal is to return the system to normal operations following the "nuclear" refactor of source code and Hydra configurations. Currently, the system is in an unstable state with failing unit tests and unverified changes. This plan addresses the immediate test failures and maps out the verification steps to ensure system integrity.

## User Review Required
> [!IMPORTANT]
> This plan modifies unit tests to match the new file structure. This is a corrective action to align tests with the refactored code.

## Proposed Changes

### Tests
#### [MODIFY] [test_architecture.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/tests/unit/test_architecture.py)
- Update patch references from `ocr.models.architecture` to `ocr.core.architecture` to reflect the move of `OCRModel` and its dependencies to the `core` package.
- Ensure all mocks target the correct namespace used by the `OCRModel` class.

## Verification Plan

### Automated Tests
1. **Unit Test**: Run the failing test to confirm the fix.
   ```bash
   pytest tests/unit/test_architecture.py -v
   ```
2. **Additional Verification**: Run `pytest` on the `tests/configs` directory (if it exists) or generally run the test suite to uncover other regressions.
   ```bash
   pytest tests/configs/
   ```

### Manual Verification
1. **Smoke Test Integration**: Verify that the refactored code (specifically `OCRModel`) can essentially run/load by invoking the help command of the main runner.
   ```bash
   python runners/train.py --help
   ```
   This confirms that Hydra can instantiate the configuration and resolves the architecture components without crashing on import errors.
