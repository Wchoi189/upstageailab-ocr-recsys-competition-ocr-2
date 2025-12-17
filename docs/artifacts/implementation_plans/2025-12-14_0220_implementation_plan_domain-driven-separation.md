---
ads_version: "1.0"
type: "implementation_plan"
category: "development"
status: "archived"
version: "1.0"
tags: ['implementation', 'plan', 'development']
title: "Domain-Driven Separation: Backend/Frontend Architecture Refactoring"
date: "2025-12-14 02:20 (KST)"
branch: "feature/agentqms-dashboard-integration"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Domain-Driven Separation: Backend/Frontend Architecture Refactoring**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

## Executive Summary

This implementation plan outlines the migration from a unified backend architecture to **Option A: Domain-Driven Separation** as recommended in the [Backend/Frontend Architecture Recommendations](../../architecture/backend-frontend-architecture-recommendations.md).

### Current State
- Single unified backend (`apps/backend/`) serves both `playground-console` and `ocr-inference-console`
- Unclear ownership of backend endpoints
- Code duplication between console API clients
- Mixed concerns: `ocr_bridge.py` vs `playground_api/` with different patterns

### Target State
- Each frontend app has its own dedicated backend service
- Shared packages for reusable code (`apps/shared/backend-shared/` and `packages/console-shared/`)
- Clear ownership: Playground Console → `apps/playground-console/backend/` (port 8001)
- Clear ownership: OCR Inference Console → `apps/ocr-inference-console/backend/` (port 8002)
- Independent deployment capability
- Consistent API versioning (`/api/v1/` prefix)

### Migration Timeline
- **Phase 1 (Weeks 1-2)**: Create shared packages, move InferenceEngine
- **Phase 2 (Weeks 3-4)**: Extract Playground Console backend
- **Phase 3 (Weeks 5-6)**: Extract OCR Inference Console backend
- **Phase 4 (Weeks 7-8)**: Deprecate unified backend

### Key Benefits
- ✅ Clear ownership: Each app team owns their full stack
- ✅ Independent evolution: Apps can evolve at different paces
- ✅ Easier onboarding: "This app = this backend" is intuitive
- ✅ Better testing: Test each app's backend in isolation
- ✅ Future-proof: Easy to extract services later if needed

---

# Living Implementation Blueprint: Domain-Driven Separation: Backend/Frontend Architecture Refactoring

## Progress Tracker
- **STATUS:** Not Started
- **CURRENT STEP:** Phase 1, Task 1.1 - Create Shared Backend Package Structure
- **LAST COMPLETED TASK:** None
- **NEXT TASK:** Create `apps/shared/backend-shared/` directory structure and move InferenceEngine

### Implementation Outline (Checklist)

#### **Phase 1: Create Shared Packages (Week 1-2)**
1. [ ] **Task 1.1: Create Shared Backend Package Structure**
   - [ ] Create `apps/shared/backend-shared/` directory structure
   - [ ] Create `apps/shared/backend-shared/inference/` for InferenceEngine
   - [ ] Create `apps/shared/backend-shared/models/` for shared Pydantic models
   - [ ] Create `apps/shared/backend-shared/utils/` for shared utilities
   - [ ] Add `__init__.py` files and package structure
   - [ ] Create `setup.py` or `pyproject.toml` for package installation

2. [ ] **Task 1.2: Move InferenceEngine to Shared Package**
   - [ ] Move `ocr/inference/engine.py` → `apps/shared/backend-shared/inference/engine.py`
   - [ ] Update imports in `apps/backend/services/ocr_bridge.py`
   - [ ] Update imports in `apps/backend/services/playground_api/routers/inference.py`
   - [ ] Update any other references to InferenceEngine
   - [ ] Verify InferenceEngine still works after move
   - [ ] Update documentation references

3. [ ] **Task 1.3: Create Shared Data Models**
   - [ ] Extract shared Pydantic models from `apps/backend/services/playground_api/routers/inference.py`
   - [ ] Create `apps/shared/backend-shared/models/inference.py` with:
     - `InferenceRequest`
     - `InferenceResponse`
     - `TextRegion`
     - `InferenceMetadata`
   - [ ] Reference data contracts from `docs/pipeline/data_contracts.md` and `docs/pipeline/inference-data-contracts.md`
   - [ ] Update both backend services to use shared models
   - [ ] Ensure models match TypeScript interfaces in `packages/console-shared/`

4. [ ] **Task 1.4: Update Shared TypeScript Package Structure**
   - [ ] Review current `packages/console-shared/` structure
   - [ ] Create `packages/console-shared/src/api/playground-client.ts`
   - [ ] Create `packages/console-shared/src/api/ocr-client.ts`
   - [ ] Create `packages/console-shared/src/types/inference.ts` with shared types
   - [ ] Create `packages/console-shared/src/types/checkpoints.ts` with checkpoint types
   - [ ] Update package exports in `packages/console-shared/src/index.ts`

#### **Phase 2: Extract Playground Console Backend (Week 3-4)**
5. [ ] **Task 2.1: Create Playground Console Backend Structure**
   - [ ] Create `apps/playground-console/backend/` directory
   - [ ] Create `apps/playground-console/backend/main.py` entry point
   - [ ] Create `apps/playground-console/backend/routers/` directory
   - [ ] Set up FastAPI app with CORS configuration
   - [ ] Configure port (default: 8001) and environment variables

6. [ ] **Task 2.2: Migrate Playground API Routers**
   - [ ] Copy `apps/backend/services/playground_api/routers/` → `apps/playground-console/backend/routers/`
   - [ ] Update imports to use shared backend package:
     - `from apps.shared.backend_shared.inference.engine import InferenceEngine`
     - `from apps.shared.backend_shared.models.inference import InferenceRequest, InferenceResponse`
   - [ ] Update router imports in `apps/playground-console/backend/main.py`
   - [ ] Update path prefixes to `/api/v1/` (e.g., `/api/v1/inference/preview`)
   - [ ] Remove dependency on `apps/backend/services/ocr_bridge.py`

7. [ ] **Task 2.3: Update Playground Console Frontend**
   - [ ] Update API client in `apps/playground-console/` to point to new backend (port 8001)
   - [ ] Update `packages/console-shared/src/api/playground-client.ts` with new base URL
   - [ ] Update environment variables and configuration
   - [ ] Test all API endpoints work with new backend
   - [ ] Update documentation in `apps/playground-console/docs/`

#### **Phase 3: Extract OCR Inference Console Backend (Week 5-6)**
8. [ ] **Task 3.1: Create OCR Inference Console Backend Structure**
   - [ ] Create `apps/ocr-inference-console/backend/` directory
   - [ ] Create `apps/ocr-inference-console/backend/main.py` entry point
   - [ ] Create `apps/ocr-inference-console/backend/routers/` directory
   - [ ] Set up FastAPI app with CORS configuration
   - [ ] Configure port (default: 8002) and environment variables

9. [ ] **Task 3.2: Migrate OCR Bridge to Dedicated Backend**
   - [ ] Extract `apps/backend/services/ocr_bridge.py` → `apps/ocr-inference-console/backend/routers/inference.py`
   - [ ] Convert `OCRBridge` class to FastAPI router endpoints
   - [ ] Update imports to use shared backend package:
     - `from apps.shared.backend_shared.inference.engine import InferenceEngine`
     - `from apps.shared.backend_shared.models.inference import InferenceRequest, InferenceResponse`
   - [ ] Update path prefixes to `/api/v1/` (e.g., `/api/v1/ocr/inference`)
   - [ ] Maintain backward compatibility with existing endpoints during migration

10. [ ] **Task 3.3: Update OCR Inference Console Frontend**
    - [ ] Update API client in `apps/ocr-inference-console/` to point to new backend (port 8002)
    - [ ] Update `packages/console-shared/src/api/ocr-client.ts` with new base URL
    - [ ] Update `apps/ocr-inference-console/src/api/ocrClient.ts` to use shared client
    - [ ] Test all API endpoints work with new backend
    - [ ] Update documentation in `apps/ocr-inference-console/docs/`

#### **Phase 4: Deprecate Unified Backend (Week 7-8)**
11. [ ] **Task 4.1: Create Transition Router**
    - [ ] Update `apps/backend/main.py` to be a simple router that delegates to both backends
    - [ ] Add deprecation warnings to all endpoints
    - [ ] Add redirects or proxy logic to route requests to appropriate backend
    - [ ] Document deprecation timeline (1 month grace period)
    - [ ] Add logging for usage tracking

12. [ ] **Task 4.2: Update Documentation**
    - [ ] Update `apps/backend/README.md` with deprecation notice
    - [ ] Update `docs/architecture/00_system_overview.md` with new architecture
    - [ ] Create migration guide for developers
    - [ ] Update API documentation with new endpoint locations
    - [ ] Update deployment documentation

13. [ ] **Task 4.3: Remove Unified Backend (After Grace Period)**
    - [ ] Verify no dependencies on `apps/backend/` remain
    - [ ] Remove `apps/backend/` directory
    - [ ] Update all references in documentation
    - [ ] Update CI/CD pipelines
    - [ ] Archive old backend code if needed

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Domain-Driven Separation: Each app owns its dedicated backend service
- [ ] Shared packages for reusable code (`apps/shared/backend-shared/` and `packages/console-shared/`)
- [ ] Consistent API versioning (`/api/v1/` prefix for all endpoints)
- [ ] Independent deployment capability for each app
- [ ] Clear module boundaries with no circular dependencies

### **Data Contracts & Models**
- [ ] Pydantic v2 models in `apps/shared/backend-shared/models/` matching TypeScript interfaces
- [ ] Compliance with data contracts from `docs/pipeline/data_contracts.md`
- [ ] Compliance with inference metadata from `docs/pipeline/inference-data-contracts.md`
- [ ] Shared types in `packages/console-shared/src/types/` synchronized with backend models
- [ ] Validation of `InferenceMetadata` with required fields: `padding_position`, `content_area`

### **Integration Points**
- [ ] InferenceEngine moved to `apps/shared/backend-shared/inference/engine.py`
- [ ] Both backends use shared InferenceEngine (no duplication)
- [ ] API clients in `packages/console-shared/src/api/` for both playground and OCR
- [ ] Path resolution using `ocr.utils.path_utils` (shared utility)
- [ ] Checkpoint discovery logic shared between backends

### **Quality Assurance**
- [ ] Unit tests for shared backend package (>80% coverage)
- [ ] Integration tests for each backend service independently
- [ ] API contract tests ensuring data format compatibility
- [ ] End-to-end tests: frontend → backend → InferenceEngine
- [ ] Performance tests: verify no regression in inference latency
- [ ] Backward compatibility tests during migration period

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] Playground Console works independently with its own backend (port 8001)
- [ ] OCR Inference Console works independently with its own backend (port 8002)
- [ ] Both consoles can run simultaneously without conflicts
- [ ] All existing API endpoints function correctly after migration
- [ ] Inference results match pre-migration output (coordinate accuracy, metadata)
- [ ] Checkpoint discovery works in both backends
- [ ] Frontend applications successfully connect to their respective backends

### **Technical Requirements**
- [ ] No code duplication: InferenceEngine exists only in shared package
- [ ] All imports updated: no references to old `apps/backend/services/` paths
- [ ] Type safety: TypeScript interfaces match Python Pydantic models
- [ ] API versioning: All endpoints use `/api/v1/` prefix consistently
- [ ] Documentation updated: Architecture docs reflect new structure
- [ ] Shared packages properly structured and importable
- [ ] No circular dependencies between packages
- [ ] Backward compatibility maintained during deprecation period

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM
### **Active Mitigation Strategies**:
1. **Incremental Migration**: Phase-by-phase approach allows rollback at any stage
2. **Parallel Running**: Keep unified backend running during migration for fallback
3. **Comprehensive Testing**: Test each phase before proceeding to next
4. **Data Contract Validation**: Use Pydantic models to catch breaking changes early
5. **Documentation First**: Update docs before code changes to clarify intent

### **Risk Assessment**:

**Risk 1: Breaking Changes During Migration**
- **Impact**: HIGH - Could break existing functionality
- **Mitigation**:
  - Maintain unified backend as fallback during migration
  - Comprehensive integration tests before each phase
  - Gradual rollout with feature flags if needed
- **Fallback**: Revert to unified backend if critical issues arise

**Risk 2: Import Path Conflicts**
- **Impact**: MEDIUM - Could cause import errors
- **Mitigation**:
  - Use absolute imports with `apps.shared.backend_shared`
  - Update all imports in single commit per phase
  - Run import validation script
- **Fallback**: Use relative imports temporarily if needed

**Risk 3: Data Contract Mismatches**
- **Impact**: HIGH - Could cause runtime errors or incorrect results
- **Mitigation**:
  - Reference `docs/pipeline/data_contracts.md` and `docs/pipeline/inference-data-contracts.md`
  - Validate Pydantic models match TypeScript interfaces
  - Test coordinate transformations match expected format
- **Fallback**: Keep existing data format during transition, add adapters if needed

**Risk 4: Deployment Complexity**
- **Impact**: MEDIUM - Multiple services to deploy
- **Mitigation**:
  - Document deployment process clearly
  - Use Docker Compose for local development
  - Provide migration scripts
- **Fallback**: Deploy as single service initially, split later

**Risk 5: Shared Package Versioning Issues**
- **Impact**: LOW - Could cause dependency conflicts
- **Mitigation**:
  - Use monorepo structure (no separate versioning needed initially)
  - Pin shared package versions if publishing separately
- **Fallback**: Keep code in apps until versioning strategy is established

---

## 🔄 **Blueprint Update Protocol**

**Update Triggers:**
- Task completion (move to next task)
- Blocker encountered (document and propose solution)
- Technical discovery (update approach if needed)
- Quality gate failure (address issues before proceeding)

**Update Format:**
1. Update Progress Tracker (STATUS, CURRENT STEP, LAST COMPLETED TASK, NEXT TASK)
2. Mark completed items with [x]
3. Add any new discoveries or changes to approach
4. Update risk assessment if needed

---

## 🚀 **Immediate Next Action**

**TASK:** Create Shared Backend Package Structure

**OBJECTIVE:** Establish the foundation for shared code by creating `apps/shared/backend-shared/` directory structure and preparing for InferenceEngine migration.

**APPROACH:**
1. Create directory structure: `apps/shared/backend-shared/inference/`, `models/`, `utils/`
2. Add `__init__.py` files to make packages importable
3. Create `setup.py` or `pyproject.toml` for package installation (if needed)
4. Document package structure and import patterns
5. Verify imports work: `from apps.shared.backend_shared import ...`

**SUCCESS CRITERIA:**
- Directory structure created and accessible
- Python can import from `apps.shared.backend_shared`
- Package structure documented
- Ready for InferenceEngine migration in next task

---

## 📚 **Reference Documentation**

### **Architecture & Design**
- [Backend/Frontend Architecture Recommendations](../../architecture/backend-frontend-architecture-recommendations.md) - Source document for Option A
- [System Overview](../../architecture/00_system_overview.md) - Current system architecture
- [API Decoupling Strategy](../../architecture/api-decoupling.md) - API design principles

### **Data Contracts**
- [Pipeline Data Contracts](../../pipeline/data_contracts.md) - Complete data contract specifications
- [Inference Data Contracts](../../pipeline/inference-data-contracts.md) - Inference-specific contracts including `InferenceMetadata`
- [OCR Inference Console Data Contracts](../../apps/ocr-inference-console/docs/data-contracts.md) - Frontend data contracts

### **Current Implementation**
- `apps/backend/services/ocr_bridge.py` - Current OCR bridge implementation
- `apps/backend/services/playground_api/` - Current playground API implementation
- `ocr/inference/engine.py` - Current InferenceEngine location (to be moved)
- `packages/console-shared/` - Current shared TypeScript package

### **Related Plans**
- [Training Refactoring Summary](../../pipeline/TRAINING_REFACTORING_SUMMARY.md) - Related refactoring context

---

## 🔍 **Key Implementation Details**

### **InferenceEngine Location**
- **Current**: `ocr/inference/engine.py`
- **Target**: `apps/shared/backend-shared/inference/engine.py`
- **Note**: InferenceEngine is already in `ocr/` package, but should be moved to shared location for domain-driven separation

### **API Endpoint Migration**
- **Playground Console**:
  - Current: `/api/inference/preview` (via unified backend)
  - Target: `http://localhost:8001/api/v1/inference/preview`
- **OCR Inference Console**:
  - Current: `/ocr/predict` (via unified backend)
  - Target: `http://localhost:8002/api/v1/ocr/inference`

### **Shared Models Reference**
Models must match contracts defined in:
- `InferenceRequest`: Base64 image, checkpoint path, optional thresholds
- `InferenceResponse`: Status, regions, processing time, metadata
- `InferenceMetadata`: **REQUIRED** fields: `padding_position`, `content_area`, `original_size`, `processed_size`, `padding`, `scale`, `coordinate_system`
- `TextRegion`: Polygon coordinates, confidence, optional text

See `docs/pipeline/inference-data-contracts.md` for complete schema.

### **Port Configuration**
- Playground Console Backend: `8001` (configurable via `PLAYGROUND_BACKEND_PORT`)
- OCR Inference Console Backend: `8002` (configurable via `OCR_BACKEND_PORT`)
- Unified Backend (deprecated): `8000` (maintained during transition)

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
