---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "active"
title: "Cleanup & Organization Implementation Plan"
date: "2026-01-08 03:40 (KST)"
version: "1.0"
tags: "cleanup, refactor, organization"
description: "Plan for implementing cleanup and organization recommendations (Processors, Scripts, Naming)."
---

# Implementation Plan - Cleanup & Organization

## Goal Description
Implement the "Future Recommendations" (1-3) from the Refactor Audit Survey to reduce bloat, improve maintainability, and enforce architectural standards. This plan excludes the Hydra Config Refactor.

## User Review Required
- [ ] **Archive Strategy**: Confirm if `aws-batch-processor` and `ocr-etl-pipeline` should be moved to a strict `archive/` directory or kept in a separate `ops/legacy` folder if not fully containerized immediately.
- [ ] **Script Consolidation**: Approval to merge `tools/` and `scripts/` into a single `ops/` directory (standardizing on "ops" for comprehensive operations including deployment/setup, or "scripts" if preferred). *Recommendation: Use `scripts/` for now to minimize disruption, but enforce subdirectories.*

## Proposed Changes

### 1. Containerize Processors
Isolate heavy, non-core components to prevent clutter in the main training repository.

#### [MOVE] `aws-batch-processor/`
- **Action**: Move to `archive/aws-batch-processor/` OR create a `docker-compose.aws-batch.yaml` and move code to `services/aws-batch/`.
- **Reason**: Recommended if not active in main loop.
- **Files**: `aws-batch-processor/**/*`

#### [MOVE] `ocr-etl-pipeline/`
- **Action**: Move to `archive/ocr-etl-pipeline/` OR `services/ocr-etl/`.
- **Reason**: Decouple ETL from core training logic.
- **Files**: `ocr-etl-pipeline/**/*`

### 2. Consolidate Scripts
Merge disjoint tool directories into a unified structure.

#### [MERGE] `tools/` and `scripts/` -> `scripts/`
- **Structure**:
    - `scripts/setup/` (environment setup, installation)
    - `scripts/data/` (dataset preparation, inspection)
    - `scripts/deploy/` (model export, serving)
    - `scripts/utils/` (general utilities)
    - `scripts/legacy/` (deprecated tools)
- **Action**: Move files from `tools/` and current `scripts/` into these subdirectories.
- **Updates**: Update CI/CD configurations (`Makefile`, GH Actions) to point to new paths.

### 3. Enforce Naming Schema
Prevent regression of the "Feature-First" architecture.

#### [NEW] `scripts/hooks/validate_architecture.py`
- **Purpose**: Verify that new files in `ocr/` follow the `ocr/<feature>/<domain>` or `ocr/core` pattern.
- **Logic**:
    - Reject direct `ocr/models/*.py` (should be `ocr/<feature>/models/` or `ocr/core/`).
    - Reject `ocr/data/*.py` (should be `ocr/<feature>/data/`).
- **Integration**: Add to `.pre-commit-config.yaml` or `Makefile` check.

## Verification Plan

### Automated Tests
- **CI Check**: Run the new architecture validation script against the current codebase to ensure it passes.
- **Path Verification**: Verify all script calls in `Makefile` and `README.md` are updated to new `scripts/` paths.

### Manual Verification
- **Structure Check**: `tree -L 2 scripts/` to verify clean organization.
- **Archive Check**: Verify `aws-batch-processor` and `ocr-etl-pipeline` are moved and don't pollute the root scope.
