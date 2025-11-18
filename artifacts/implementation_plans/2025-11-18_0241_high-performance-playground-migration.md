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
- **CURRENT STEP:** Phase 1 - SPA Scaffold & Command Builder Parity (NEARLY COMPLETE)
- **LAST COMPLETED TASK:** Phase 1, Task 1.4 - Built Command Execution Drawer with download, validation status, and execution log placeholder
- **NEXT TASK:** Phase 2 - Worker Pipeline & Preprocessing Studio (or wrap up Phase 1)

### Blockers & Open Issues

- **FastAPI startup latency** (Phase 1 regression): importing `ui.utils.config_parser` and the command builder services now triggers Streamlit + registry initialization, which adds a ~10–15 second cold-start before `uvicorn` begins listening. Until this is optimized the SPA shows a spinner/timeout unless the API is pre-warmed. A diagnostic helper (`test_api_startup.py`) was added, but we still need to (a) trim the import graph, (b) cache heavy metadata, or (c) replace the Streamlit-era dependencies with lightweight JSON manifests before Phase 2.

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
1. [x] **Task 1.1: Vite + React SPA Setup**
   - [x] Initialize Vite project in `frontend/` with React + TypeScript
   - [x] Configure routing (React Router) for unified shell vs. micro-apps
   - [x] Set up API client layer with typed fetch wrappers
   - [x] Connect to FastAPI backend at `http://127.0.0.1:8000`
   - [x] Verify `/api/commands/schemas` endpoint returns schema metadata
   - [x] Configure ESLint/Prettier with 100-char line length
   - [x] Configure Ruff linter with 140-char line length
   - [x] Fixed router import paths (changed `...utils` to `..utils`)
   - [x] Added uvicorn dependency and tested full stack connectivity

2. [x] **Task 1.2: Command Console Module**
   - [x] Build schema-driven form generator (mirrors `ui.utils.ui_generator`)
   - [x] Implement training/test/predict tabs with shared form primitives
   - [x] Wire `/api/commands/build` for command generation
   - [x] Add command diff viewer (before/after editing)
   - [x] Integrate validation error display
   - [x] Created `/api/commands/schemas/{schema_id}` endpoint with dynamic options
   - [x] Built FormPrimitives components (TextInput, Checkbox, SelectBox, etc.)
   - [x] Implemented SchemaForm with conditional visibility support
   - [x] Added CommandDisplay and CommandDiffViewer components
   - [x] All code follows 100-char line length and explicit typing standards

3. [x] **Task 1.3: Recommendation Panel**
   - [x] Surface use case recommendations via `/api/commands/recommendations`
   - [x] Build recommendation card UI component
   - [x] Wire recommendation selection to form state
   - [x] Add architecture-aware metadata tooltips
   - [x] Created RecommendationCard and RecommendationsGrid components
   - [x] Implemented Tooltip and InfoIcon components
   - [x] Integrated recommendations with architecture filter auto-reload
   - [x] Added collapsible recommendations panel
   - [x] Tooltips integrated into all form primitives

4. [x] **Task 1.4: Command Execution Drawer**
   - [x] Create command preview panel (read-only, copy button)
   - [x] Add execution log console (streaming output placeholder)
   - [x] Implement download command as file
   - [x] Enhanced CommandDisplay with validation status badge
   - [x] Added download functionality (saves as command.sh)
   - [x] Created ExecutionLog component with log levels and timestamps
   - [x] Disabled state for buttons when no command
   - [x] Visual feedback for all actions (copy, download)

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

### **Code Quality & Standards**
- [x] Coding standards document created (`docs/CODING_STANDARDS.md`)
- [ ] All Python code follows 140 char line length, type hints, < 50 line functions
- [ ] All TypeScript code follows 100 char line length, explicit types, < 40 line functions
- [x] Linters/formatters configured and passing (Ruff for Python, ESLint/Prettier for TS)

### **Architecture & Design**
- [x] Modular SPA architecture (Vite + React) with optional module federation
- [x] FastAPI backend with stateless service layer
- [x] Web Worker pipeline for client-side transforms
- [x] Hybrid execution contract (client-first, backend fallback)
- [ ] TypeScript schema library (Zod) mirroring Python dataclasses

### **Integration Points**
- [x] FastAPI endpoints for command builder, inference, evaluation, pipelines
- [x] Worker RPC interface (Comlink/MessageChannel)
- [x] React Router for unified shell vs. micro-apps
- [x] API client layer with typed fetch wrappers

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

**TASK:** Phase 1, Task 1.2 - Build Command Console Module with schema-driven form generator

**OBJECTIVE:** Build the command builder UI that generates CLI commands from user input, matching the functionality of the Streamlit command builder.

**APPROACH:**
1. Create schema-driven form generator component that reads from `/api/commands/schemas`
2. Implement training/test/predict tabs with shared form primitives
3. Wire `/api/commands/build` endpoint to generate commands from form state
4. Add command diff viewer to show before/after editing
5. Integrate validation error display from API responses
6. Ensure form state management works correctly with React state or Zustand

**SUCCESS CRITERIA:**
- Form generator dynamically creates inputs based on schema metadata
- Training/test/predict tabs switch correctly and maintain separate form state
- Generated commands match Streamlit baseline (99%+ parity)
- Validation errors display clearly to users
- Command diff viewer shows changes accurately

**STOP CONDITIONS:**
- If schema parsing fails, verify API endpoint returns correct format
- If form generation breaks, check schema structure matches expected format
- If command generation differs from Streamlit, document differences and investigate

---

## 📝 **Web Worker Implementation Prompt**

**For autonomous web worker agents executing this plan:**

You are implementing a high-performance image processing playground inspired by Albumentations and the Upstage Document OCR console. Your current focus is **Phase 1, Task 1.2** (Command Console Module).

**Context:**
- FastAPI backend is ready at `services/playground_api/` with endpoints for commands, inference, pipelines, and evaluation
- Worker prototypes exist in `frontend/workers/` but need integration into the SPA
- Design system and architecture decisions are documented in `docs/adr/ADR-frontend-stack.md` and `docs/ui/design-system.md`
- Migration roadmap is in `docs/ui/migration-roadmap.md`

**Coding Standards (MUST FOLLOW):**
- **TypeScript**: 100 character line length, explicit types required, functions < 40 lines (target), files < 300 lines (target)
- **Python**: 140 character line length, type hints required, functions < 50 lines (target), files < 500 lines (target)
- See `docs/CODING_STANDARDS.md` for complete guidelines including naming conventions, documentation, and refactoring strategies

**Your immediate task:**
1. Build schema-driven form generator component (mirrors `ui.utils.ui_generator`)
2. Implement training/test/predict tabs with shared form primitives
3. Wire `/api/commands/build` endpoint for command generation
4. Add command diff viewer (before/after editing)
5. Integrate validation error display
6. **Ensure all code follows the coding standards** - run linters/formatters before committing

**Key files to reference:**
- `docs/CODING_STANDARDS.md` - **CRITICAL**: Follow coding standards for line length, function/file size, naming conventions, and type safety
- `docs/adr/ADR-frontend-stack.md` - Stack decisions and routing strategy
- `docs/ui/design-system.md` - Component specifications
- `services/playground_api/app.py` - API endpoint structure
- `frontend/workers/types.ts` - Worker contract definitions

**When you complete this task:**
- Update the Progress Tracker in this document
- Mark Task 1.2 as complete
- Move to Task 1.3 (Recommendation Panel)
- Report any blockers or deviations from the plan

**Remember:** Always update the Progress Tracker after each task. The NEXT TASK field tells you exactly what to work on next.

---

*This implementation plan follows the Blueprint Protocol Template (PROTO-GOV-003) for systematic, autonomous execution with clear progress tracking.*

