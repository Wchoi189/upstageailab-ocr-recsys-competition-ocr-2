---
title: "High-Performance Playground Migration - Web Worker Implementation"
author: "ai-agent"
timestamp: "2025-11-18 02:41 KST"
branch: "main"
type: "implementation_plan"
category: "ui"
status: "in-progress"
tags: ["implementation_plan", "ui", "playground", "web-workers", "migration", "spa"]
---

# Master Prompt

You are an autonomous AI agent, my Chief of Staff for implementing the **High-Performance Playground Migration - Web Worker Implementation**. Your primary responsibility is to execute the "Living Implementation Blueprint" systematically, handle outcomes, and keep track of our progress. Do not ask for clarification on what to do next; your next task is always explicitly defined.

---

**Your Core Workflow is a Goal-Execute-Update Loop:**
1. **Goal:** A clear `🎯 Goal` will be provided for you to achieve.
2. **Execute:** You will start working on the task defined in the `NEXT TASK`
3. **Handle Outcome & Update:** Based on the success or failure of the command, you will follow the specified contingency plan. Your response must be in two parts:
   * **Part 1: Execution Report:** Provide a concise summary of the results and analysis of the outcome (e.g., "All tests passed" or "Test X failed due to an IndexError...").
   * **Part 2: Blueprint Update Confirmation:** Confirm that the living blueprint has been updated with the new progress status and next task. The updated blueprint is available in the workspace file.

---

# Living Implementation Blueprint: High-Performance Playground Migration

## Progress Tracker
**⚠️ CRITICAL: This Progress Tracker MUST be updated after each task completion, blocker encounter, or technical discovery. Required for iterative debugging and incremental progress tracking.**

- **STATUS:** In Progress
- **CURRENT STEP:** Phase 0 - Foundation Complete, Starting Phase 1
- **LAST COMPLETED TASK:** Created FastAPI backend stubs, worker TypeScript prototypes, parity matrix, ADR, design system docs, and migration roadmap. Added `run_spa.py` for dev server orchestration.
- **NEXT TASK:** Phase 1, Task 1.1 - Set up Vite + React SPA scaffold with routing and connect to FastAPI backend

### Implementation Outline (Checklist)

#### **Phase 0: Foundation & Planning (COMPLETED ✅)**
1. [x] **Task 0.1: Parity Inventory & Service Extraction**
   - [x] Documented feature parity matrix (`docs/ui/parity.md`)
   - [x] Identified shared business logic extraction targets
   - [x] Created FastAPI service stubs (`services/playground_api/`)

2. [x] **Task 0.2: Architecture & Design Documentation**
   - [x] Created ADR for frontend stack (`docs/adr/ADR-frontend-stack.md`)
   - [x] Defined design system spec (`docs/ui/design-system.md`)
   - [x] Documented worker utilization blueprint (`docs/ui/worker-blueprint.md`)

3. [x] **Task 0.3: Backend API Contracts**
   - [x] Implemented `/api/commands/*` router (command builder endpoints)
   - [x] Implemented `/api/inference/*` router (checkpoint discovery, preview stubs)
   - [x] Implemented `/api/pipelines/*` router (preview/fallback contracts)
   - [x] Implemented `/api/evaluation/*` router (comparison presets)
   - [x] Documented pipeline contract (`docs/api/pipeline-contract.md`)

4. [x] **Task 0.4: Worker Pipeline Prototypes**
   - [x] Created TypeScript worker types (`frontend/workers/types.ts`)
   - [x] Implemented pipeline worker (`frontend/workers/pipelineWorker.ts`)
   - [x] Created transform handlers (`frontend/workers/transforms.ts`)
   - [x] Added rembg worker stub (`frontend/workers/rembgWorker.ts`)

5. [x] **Task 0.5: Testing & Tooling Infrastructure**
   - [x] Created dataset sampler script (`scripts/datasets/sample_images.py`)
   - [x] Added performance benchmark harness (`tests/perf/pipeline_bench.py`)
   - [x] Created `run_spa.py` dev server runner
   - [x] Added FastAPI/Pydantic dependencies to `pyproject.toml`

6. [x] **Task 0.6: Migration Planning**
   - [x] Created migration roadmap (`docs/ui/migration-roadmap.md`)
   - [x] Documented testing/observability strategy (`docs/ui/testing-observability.md`)
   - [x] Added beta CTA banners to Streamlit apps

#### **Phase 1: SPA Scaffold & Command Builder Parity (IN PROGRESS)**
1. [ ] **Task 1.1: Vite + React SPA Setup**
   - [ ] Initialize Vite project in `frontend/` with React + TypeScript
   - [ ] Configure routing (React Router) for unified shell vs. micro-apps
   - [ ] Set up API client layer with typed fetch wrappers
   - [ ] Connect to FastAPI backend at `http://127.0.0.1:8000`
   - [ ] Verify `/api/commands/schemas` endpoint returns schema metadata

2. [ ] **Task 1.2: Command Console Module**
   - [ ] Build schema-driven form generator (mirrors `ui.utils.ui_generator`)
   - [ ] Implement training/test/predict tabs with shared form primitives
   - [ ] Wire `/api/commands/build` for command generation
   - [ ] Add command diff viewer (before/after editing)
   - [ ] Integrate validation error display

3. [ ] **Task 1.3: Recommendation Panel**
   - [ ] Surface use case recommendations via `/api/commands/recommendations`
   - [ ] Build recommendation card UI component
   - [ ] Wire recommendation selection to form state
   - [ ] Add architecture-aware metadata tooltips

4. [ ] **Task 1.4: Command Execution Drawer**
   - [ ] Create command preview panel (read-only, copy button)
   - [ ] Add execution log console (streaming output placeholder)
   - [ ] Implement download command as file
   - [ ] Validate generated commands match Streamlit baseline (diff script)

#### **Phase 2: Worker Pipeline & Preprocessing Studio (PENDING)**
1. [ ] **Task 2.1: Worker Pool Manager**
   - [ ] Implement `workerHost.ts` with dynamic pool sizing
   - [ ] Add priority queue for task scheduling
   - [ ] Wire cancellation tokens for slider spam handling
   - [ ] Create `useWorkerTask` React hook

2. [ ] **Task 2.2: Preprocessing Canvas Component**
   - [ ] Build before/after image viewer with side-by-side layout
   - [ ] Implement parameter control tray (sliders, toggles)
   - [ ] Connect controls to worker pipeline via `useWorkerTask`
   - [ ] Add debouncing (75ms) for slider updates

3. [ ] **Task 2.3: rembg Hybrid Routing**
   - [ ] Bundle ONNX.js rembg model (~3 MB) for client-side
   - [ ] Implement fallback heuristics (image size, latency thresholds)
   - [ ] Wire `/api/pipelines/fallback` for backend rembg
   - [ ] Add cache key generation `(imageHash, paramsHash)`

4. [ ] **Task 2.4: Performance Validation**
   - [ ] Run `tests/perf/pipeline_bench.py` with dataset samples
   - [ ] Verify <100ms for contrast/blur, <400ms for client rembg
   - [ ] Test worker queue depth during slider spam (Playwright)
   - [ ] Document any latency regressions

#### **Phase 3: Inference & Comparison Studios (PENDING)**
1. [ ] **Task 3.1: Inference Checkpoint Picker**
   - [ ] Build checkpoint catalog UI using `/api/inference/checkpoints`
   - [ ] Add search/filter by architecture, backbone, epochs
   - [ ] Display checkpoint metadata (hmean, created date)

2. [ ] **Task 3.2: Inference Preview Canvas**
   - [ ] Create polygon overlay renderer for detected text regions
   - [ ] Wire `/api/inference/preview` for single image inference
   - [ ] Add hyperparameter controls (confidence threshold, etc.)
   - [ ] Implement batch job submission UI

3. [ ] **Task 3.3: Comparison Studio**
   - [ ] Build parameter sweep configuration UI
   - [ ] Wire `/api/evaluation/compare` for multi-config runs
   - [ ] Create side-by-side results comparison view
   - [ ] Add metrics visualization (charts, tables)

4. [ ] **Task 3.4: Gallery Component**
   - [ ] Implement responsive masonry grid for image gallery
   - [ ] Add lazy loading with IntersectionObserver
   - [ ] Wire dataset sampling via `/api/evaluation/gallery-root`
   - [ ] Add image selection for batch operations

#### **Phase 4: Testing, Observability & Rollout (PENDING)**
1. [ ] **Task 4.1: Playwright Test Suite**
   - [ ] Create E2E tests for command builder flow
   - [ ] Add slider spam test for worker queue stability
   - [ ] Test rembg fallback routing
   - [ ] Validate command generation parity (99%+ match)

2. [ ] **Task 4.2: Telemetry Integration**
   - [ ] Implement `/api/metrics` endpoint for worker events
   - [ ] Add worker lifecycle event logging
   - [ ] Create telemetry dashboard component
   - [ ] Track cache hit rates, fallback frequency

3. [ ] **Task 4.3: Documentation & Handoff**
   - [ ] Update README with `run_spa.py` instructions
   - [ ] Document worker debugging procedures
   - [ ] Create developer onboarding guide
   - [ ] Archive Streamlit apps (Phase 4 exit criteria)

---

## 📋 **Technical Requirements Checklist**

### **Architecture & Design**
- [x] Modular SPA architecture (Vite + React) with optional module federation
- [x] FastAPI backend with stateless service layer
- [x] Web Worker pipeline for client-side transforms
- [x] Hybrid execution contract (client-first, backend fallback)
- [ ] TypeScript schema library (Zod) mirroring Python dataclasses

### **Integration Points**
- [x] FastAPI endpoints for command builder, inference, evaluation, pipelines
- [x] Worker RPC interface (Comlink/MessageChannel)
- [ ] React Router for unified shell vs. micro-apps
- [ ] API client layer with typed fetch wrappers

### **Quality Assurance**
- [ ] Unit test coverage >80% for worker transforms
- [ ] Integration tests for API endpoints
- [ ] Performance benchmarks (<100ms preview latency)
- [ ] Playwright E2E suite for critical user flows

---

## 🎯 **Success Criteria Validation**

### **Functional Requirements**
- [ ] Command Builder generates identical CLI commands vs. Streamlit (99%+ parity)
- [ ] Preprocessing previews render in <100ms (contrast) / <400ms (rembg client)
- [ ] Worker queue depth <5 during slider spam test
- [ ] rembg fallback routes to backend for >2048px images

### **Technical Requirements**
- [ ] All TypeScript code is type-checked and linted
- [ ] FastAPI endpoints return proper HTTP status codes
- [ ] Worker memory usage <100MB per worker instance
- [ ] SPA bundle size <2MB (gzipped) for initial load

---

## 📊 **Risk Mitigation & Fallbacks**

### **Current Risk Level**: MEDIUM
### **Active Mitigation Strategies**:
1. **Incremental Development**: Phase-by-phase rollout with dual-runway (Streamlit + SPA)
2. **Performance Monitoring**: Real-time telemetry for worker queue depth and latency
3. **Fallback Routing**: Automatic backend escalation when client workers saturate
4. **Parity Validation**: Automated diff scripts to ensure command generation accuracy

### **Fallback Options**:
1. **Worker Instability**: Disable client-side rembg, route all to backend
2. **Performance Regressions**: Throttle worker pool size, increase debounce delays
3. **Bundle Size Issues**: Code-split rembg ONNX model, lazy-load on demand
4. **API Contract Drift**: Version API endpoints, maintain backward compatibility

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

**TASK:** Phase 1, Task 1.1 - Set up Vite + React SPA scaffold with routing and connect to FastAPI backend

**OBJECTIVE:** Create the foundational SPA structure that will host all playground modules, with proper routing and API integration.

**APPROACH:**
1. Initialize Vite project in `frontend/` directory with React + TypeScript template
2. Install dependencies: `react-router-dom`, `zustand` (state management), `axios` or `fetch` wrapper
3. Configure Vite proxy to forward `/api/*` requests to `http://127.0.0.1:8000`
4. Set up basic routing structure (home, command-builder, preprocessing, inference, comparison)
5. Create API client module with typed fetch wrappers for all FastAPI endpoints
6. Add a simple test page that calls `/api/commands/schemas` to verify connectivity
7. Update `run_spa.py` to optionally start Vite dev server alongside FastAPI

**SUCCESS CRITERIA:**
- `npm run dev` starts Vite server on `http://localhost:5173`
- SPA can successfully fetch from FastAPI at `http://127.0.0.1:8000/api/commands/schemas`
- Basic routing works (can navigate between placeholder pages)
- TypeScript compilation passes without errors

**STOP CONDITIONS:**
- If Vite setup fails due to dependency conflicts, document the issue and escalate
- If API connectivity fails, verify FastAPI server is running and CORS is configured
- If routing breaks, check React Router version compatibility

---

## 📝 **Web Worker Implementation Prompt**

**For autonomous web worker agents executing this plan:**

You are implementing a high-performance image processing playground inspired by Albumentations and the Upstage Document OCR console. Your current focus is **Phase 1, Task 1.1** (SPA scaffold setup).

**Context:**
- FastAPI backend is ready at `services/playground_api/` with endpoints for commands, inference, pipelines, and evaluation
- Worker prototypes exist in `frontend/workers/` but need integration into the SPA
- Design system and architecture decisions are documented in `docs/adr/ADR-frontend-stack.md` and `docs/ui/design-system.md`
- Migration roadmap is in `docs/ui/migration-roadmap.md`

**Your immediate task:**
1. Navigate to the project root
2. Run `npm create vite@latest frontend -- --template react-ts` (or equivalent)
3. Install required dependencies (see ADR for full list)
4. Configure Vite proxy for API calls
5. Set up React Router with routes matching the unified app structure
6. Create a minimal API client that calls `/api/commands/schemas`
7. Verify the setup by running both `python run_spa.py --api-only` and `npm run dev` in `frontend/`

**Key files to reference:**
- `docs/adr/ADR-frontend-stack.md` - Stack decisions and routing strategy
- `docs/ui/design-system.md` - Component specifications
- `services/playground_api/app.py` - API endpoint structure
- `frontend/workers/types.ts` - Worker contract definitions

**When you complete this task:**
- Update the Progress Tracker in this document
- Mark Task 1.1 as complete
- Move to Task 1.2 (Command Console Module)
- Report any blockers or deviations from the plan

**Remember:** Always update the Progress Tracker after each task. The NEXT TASK field tells you exactly what to work on next.

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*

