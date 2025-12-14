---
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'development']
title: "Complete domain-driven backend reconstruction"
date: "2025-12-14 17:46 (KST)"
branch: "main"
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **Complete domain-driven backend reconstruction**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: Complete domain-driven backend reconstruction

## Progress Tracker
- **STATUS:** In Progress
- **CURRENT STEP:** Phase 1, Task 1.2 - Implement shared inference primitives
- **LAST COMPLETED TASK:** Task 1.1 - Inventory archive and define shared backend contract (COMPLETED 2025-12-14)
- **NEXT TASK:** Implement shared backend package (create __init__.py files, re-export InferenceEngine, create Pydantic models)

### Implementation Outline (Checklist)

#### **Phase 1: Foundation (Shared Components)**
1. [x] **Task 1.1: Recover shared backend module layout**
   - [x] Read archive manifest and code refs in docs/archive/archive_code/deprecated/apps-backend/
   - [x] Define folder structure for apps/shared/backend_shared (inference/, models/, config/)
   - [x] Enumerate required interfaces (InferenceEngine, model loading, prediction DTOs)
   - [x] Document shared package contract in docs/artifacts/specs/shared-backend-contract.md

2. [ ] **Task 1.2: Implement shared inference primitives**
   - [ ] Port/migrate InferenceEngine from archived unified backend into apps/shared/backend_shared/inference/engine.py
   - [ ] Add config/loader utilities if needed (checkpoint path resolution)
   - [ ] Add Pydantic v2 models in apps/shared/backend_shared/models/inference.py matching frontend types

3. [ ] **Task 1.3: Publish shared package contract**
   - [ ] Add __init__.py exports and README with usage examples
   - [ ] Add minimal unit smoke test for engine init + dummy predict (can be skipped but listed)
   - [ ] Ensure import paths used by app backends resolve

#### **Phase 2: OCR Backend (Port 8002)**
4. [ ] **Task 2.1: Scaffold routers and dependencies**
   - [ ] Create routers/health.py, routers/checkpoints.py, routers/inference.py using shared models
   - [ ] Add dependency wiring for InferenceEngine (startup lifespan)
   - [ ] Add settings (env vars: OCR_CHECKPOINT_PATH, MODEL_DEVICE, etc.)

5. [ ] **Task 2.2: Implement inference + checkpoints**
   - [ ] Implement checkpoint discovery with metadata (size, mtime, display name)
   - [ ] Implement /api/v1/inference/preview (base64 image -> predictions via shared engine)
   - [ ] Implement /api/v1/health returning checkpoint status and model device

6. [ ] **Task 2.3: Wire frontend config**
   - [ ] Ensure Makefile targets set VITE_API_URL=http://127.0.0.1:8002/api/v1
   - [ ] Align frontend API paths with v1 routes (update ocrClient if needed)
   - [ ] Add README snippet for local dev startup

#### **Phase 3: Playground Backend (Port 8001)**
7. [ ] **Task 3.1: Recreate playground backend skeleton**
   - [ ] Define routers for inference, commands, evaluation, checkpoints
   - [ ] Wire shared engine and config (dataset/mode handling if applicable)
   - [ ] Add health endpoint and settings

8. [ ] **Task 3.2: Implement key endpoints**
   - [ ] /api/v1/inference/preview and checkpoints parity with OCR (playground needs)
   - [ ] Commands/evaluation endpoints restored per archive references
   - [ ] Update Next.js console API client base URL if required

#### **Phase 4: Testing & Validation**
9. [ ] **Task 4.1: Backend validation**
   - [ ] Run uvicorn smoke tests for both backends (no port conflicts)
   - [ ] Add basic pytest covering checkpoint listing + health
   - [ ] Confirm CORS configuration for Vite/Next dev hosts

10. [ ] **Task 4.2: Frontend integration checks**
    - [ ] Verify OCR console lists checkpoints and runs inference
    - [ ] Verify Playground console connects and loads modes/checkpoints
    - [ ] Update docs/guides with final commands and endpoints

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [ ] Shared inference engine lives in apps/shared/backend_shared/inference/engine.py
- [ ] Pydantic v2 models in apps/shared/backend_shared/models/inference.py align with TS client types
- [ ] FastAPI apps expose versioned routes under /api/v1
- [ ] Config driven via env vars (OCR_CHECKPOINT_PATH, BACKEND_HOST/PORT, MODEL_DEVICE)

### **Integration Points**
- [ ] OCR console frontend uses VITE_API_URL=http://127.0.0.1:8002/api/v1
- [ ] Playground console frontend points to http://127.0.0.1:8001/api/v1
- [ ] Makefile targets start both backends without port conflicts and set env defaults
- [ ] Checkpoint discovery reads outputs/experiments/train/ocr (and other runs as needed)

### **Quality Assurance**
- [ ] Smoke pytest for health + checkpoints per backend
- [ ] Manual integration check: OCR console lists checkpoints and returns inference payload
- [ ] Optional load/perf smoke: single-image inference completes < 2s on dev hardware
- [ ] Documentation updated to reflect commands and endpoints

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] OCR backend serves /api/v1/inference/checkpoints with real data from outputs/experiments/train/ocr
- [ ] OCR backend serves /api/v1/inference/preview returning predictions + meta
- [ ] Playground backend serves /api/v1 endpoints (inference, checkpoints, commands/eval where applicable)
- [ ] Frontends load checkpoints and complete an inference round-trip in dev mode

### **Technical Requirements**
- [ ] Code is type-hinted and lint-clean (ruff) with Pydantic v2 models
- [ ] CORS permits local dev origins (5173, 3000) while allowing lockdown for prod
- [ ] No port conflicts (8001, 8002) and Makefile utilities can kill/restart cleanly
- [ ] README/docs updated to reflect new startup commands and env vars

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: LOW
### **Active Mitigation Strategies**:
1. Incremental restoration: shared → OCR backend → Playground backend, validating each stage
2. Keep placeholder inference responses until engine wiring is verified
3. Use Makefile kill-ports and distinct hosts to avoid port conflicts during dev
4. **COMPLETED**: Shared backend contract documented (reduces implementation risk)

### **Fallback Options**:
1. If engine porting blocks progress, expose file-based mock predictions to unblock UI
2. If checkpoint discovery is slow, use precomputed metadata or limit depth
3. If Next.js/OCR Vite integration lags, document curl-based backend verification as interim

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

**TASK:** Implement shared backend package (inference engine + Pydantic models)

**OBJECTIVE:** Create the shared backend package implementation with InferenceEngine re-export and Pydantic v2 models matching the documented contract.

**APPROACH:**
1. Create `apps/shared/backend_shared/__init__.py` with package-level exports
2. Create `apps/shared/backend_shared/inference/__init__.py` and re-export InferenceEngine from ocr.inference.engine
3. Create `apps/shared/backend_shared/models/inference.py` with all Pydantic v2 models (InferenceRequest, InferenceResponse, TextRegion, InferenceMetadata, Padding)
4. Create `apps/shared/backend_shared/README.md` with usage examples
5. Test imports to ensure all paths resolve correctly

**SUCCESS CRITERIA:**
- All imports from shared package work correctly
- InferenceEngine can be imported from apps.shared.backend_shared.inference
- All Pydantic models can be imported from apps.shared.backend_shared.models.inference
- No circular import issues
- Code is type-hinted and lint-clean

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*
