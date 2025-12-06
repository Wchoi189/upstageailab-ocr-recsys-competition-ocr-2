---
title: "2025 12 03 Padding Orientation Migration"
date: "2025-12-06 18:08 (KST)"
type: "implementation_plan"
category: "planning"
status: "active"
version: "1.0"
tags: ['implementation_plan', 'planning', 'documentation']
---





# Padding Orientation Migration Plan

**Note**: This file should be renamed to `YYYY-MM-DD_HHMM_implementation_plan_padding-orientation-migration.md` with actual timestamp.
**AgentQMS Workflow**: Use `python AgentQMS/agent_tools/core/artifact_workflow.py create --type implementation_plan --name padding-orientation-migration --title "Padding Orientation Migration Plan"` if creating new artifact.

**Date**: 2025-12-03 HH:MM
**Status**: Planning
**Related**: BUG-002 (visual padding mismatch)

## Overview

Migrate from top-left to centered padding while maintaining backward compatibility. Requires updates across preprocessing, postprocessing, coordinate transformations, training configs, and frontend rendering.

## Progress Tracking

### Phase Status

- [ ] **Phase 1**: Data Contract Foundation
- [ ] **Phase 2**: Preprocessing Updates
- [ ] **Phase 3**: Postprocessing Updates
- [ ] **Phase 4**: Engine Orchestration
- [ ] **Phase 5**: Training Config Updates
- [ ] **Phase 6**: Frontend Updates
- [ ] **Phase 7**: Testing & Validation
- [ ] **Phase 8**: Documentation & Migration Guide

### Completed Work

_Update this section as phases complete:_

```
Phase 1: [YYYY-MM-DD HH:MM] - Data contracts updated
Phase 2: [YYYY-MM-DD HH:MM] - Preprocessing supports both padding positions
...
```

### Blockers & Decisions

_Record blockers and key decisions here:_

- [Decision] Maintain backward compatibility - support both padding positions
- [Decision] Full pipeline scope - including training transforms
- [Blocker] [Description] - [Resolution/Status]

### Test Results

_Record test results as phases complete:_

```
Phase 1 Tests: [Status] - [Notes]
Phase 2 Tests: [Status] - [Notes]
...
```

## Implementation Phases

### Phase 1: Data Contract Foundation

**Files** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/backend/services/playground_api/routers/inference.py` - Add `padding_position`, `content_area` to `InferenceMetadata`
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/frontend/src/api/inference.ts` - Mirror TypeScript interfaces
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/pipeline/inference-data-contracts.md` - Contract documentation

**Deliverables**:
- Updated Pydantic models with validation
- Updated TypeScript interfaces
- Contract documentation

### Phase 2: Preprocessing Updates

**Files** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ui/utils/inference/preprocess.py` - Add `padding_position` parameter, implement centered padding
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ui/utils/inference/config_loader.py` - Add padding position to `PreprocessSettings`

**Key Changes**:
- Centered padding calculation: `pad_left = pad_w // 2`, `pad_right = pad_w - pad_left`
- Content area calculation for metadata

### Phase 3: Postprocessing Updates

**Files** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ui/utils/inference/postprocess.py` - Update `compute_inverse_matrix()` with translation for centered padding
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr/utils/geometry_utils.py` - Verify center padding support

**Key Changes**:
- Translation components for centered padding: `translation_x = -pad_left * inv_scale`

### Phase 4: Engine Orchestration

**Files** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ui/utils/inference/engine.py` - Pass `padding_position` through pipeline, update coordinate mapping

**Key Changes**:
- Update `_map_polygons_to_preview_space()` with translation offsets
- Metadata generation includes `padding_position` and `content_area`

### Phase 5: Training Config Updates

**Files** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/transforms/base.yaml` - Add `padding_position` parameter to all transforms

**Migration Strategy**:
- Default to top-left for backward compatibility
- Add config override for centered padding

### Phase 6: Frontend Updates

**Files** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/frontend/src/components/inference/InferencePreviewCanvas.tsx` - Update `drawPolygon()` to handle padding position

**Key Changes**:
- Use `meta.padding_position` for coordinate handling
- Adjust offsets based on content area

### Phase 7: Testing & Validation

**Test Files** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/tests/unit/test_geometry_utils_coordinate_transformation.py` - Update for both padding positions
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/tests/unit/test_preprocessing_contracts.py` - Add padding position tests
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/tests/integration/test_inference_service.py` - Integration tests

### Phase 8: Documentation & Migration Guide

**Deliverables**:
- API migration guide
- Coordinate transformation documentation
- Data contract specifications
- Training config migration instructions

## Key Files Reference

**Backend** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ui/utils/inference/preprocess.py` - Padding application
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ui/utils/inference/postprocess.py` - Inverse coordinate mapping
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/ui/utils/inference/engine.py` - Pipeline orchestration
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/backend/services/playground_api/routers/inference.py` - API contracts

**Frontend** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/frontend/src/components/inference/InferencePreviewCanvas.tsx` - Rendering
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/frontend/src/api/inference.ts` - TypeScript contracts

**Config** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/configs/transforms/base.yaml` - Training transforms

**Tests** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/tests/unit/test_geometry_utils_coordinate_transformation.py` - Coordinate tests
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/tests/unit/test_preprocessing_contracts.py` - Contract tests

## Data Contracts

**Updated Contracts** (absolute paths):
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/pipeline/inference-data-contracts.md` - NEW: Added `padding_position` and `content_area`
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/pipeline/data_contracts.md` - Updated Inference Engine section
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/pipeline/README.md` - Contract index and cross-references
- `/workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/pipeline/DATA_CONTRACT_ORGANIZATION.md` - Contract organization guidelines

**Missing Contracts Identified**:
- ✅ `padding_position` - Now defined in InferenceMetadata
- ✅ `content_area` - Now defined in InferenceMetadata
- ✅ Coordinate transformation contracts - Now documented in inference-data-contracts.md

## Session Resume Prompt

```
I'm resuming work on the padding orientation migration plan.

Last completed: [Phase X - YYYY-MM-DD HH:MM]
Current phase: [Phase Y]
Blockers: [List any blockers]

Please:
1. Review progress tracking section above
2. Continue with [Phase Y] implementation
3. Update progress tracking as work completes (use YYYY-MM-DD HH:MM format)
4. Document any blockers or decisions in the plan

Plan location: /workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/artifacts/implementation_plans/2025-12-03_HHMM_implementation_plan_padding-orientation-migration.md

Note: Filename should be updated to include actual HHMM timestamp when creating/updating.
```

## Success Criteria

1. Both top-left and centered padding work correctly
2. All coordinate transformations are accurate (roundtrip tests pass)
3. Data contracts are comprehensive and validated
4. Backward compatibility maintained
5. Visual alignment verified
6. Full test coverage
7. Documentation complete
