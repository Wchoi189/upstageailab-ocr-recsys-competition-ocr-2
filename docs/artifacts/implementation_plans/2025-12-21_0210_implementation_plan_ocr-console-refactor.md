---
title: "OCR Console Refactoring Implementation"
date: "2025-12-21 02:10 (KST)"
type: "implementation_plan"
category: "planning"
status: "completed"
version: "1.0"
ads_version: "1.0"
completion_date: "2025-12-21 03:30"
assessment_ref: "brain/8fb83580-aa28-44b5-93f5-f940df97ce55/refactoring-assessment.md"
related_contracts: ""
tags: "["ocr-console", "refactoring", "checkpoint-service", "frontend-context"]"
implementation_summary: ""
files_created: "7"
files_modified: "4"
backend_loc_reduced: "150"
frontend_props_eliminated: "41"
verification: "backend_health_ok|frontend_build_ok|e2e_test_ok"
---






# OCR Console Refactoring Implementation

## Goal

Refactor OCR inference console to improve:
- AI tool effectiveness (smaller, focused modules)
- Startup performance (async checkpoint loading)
- Maintainability (service extraction, reduced prop drilling)
- Debugging (structured errors, request tracing)

## Reference Documents

| Document | Path |
|----------|------|
| Refactoring Assessment | `brain/.../refactoring-assessment.md` |
| Shared Backend Contract | `docs/artifacts/specs/shared-backend-contract.md` |
| API Data Contracts | `apps/ocr-inference-console/docs/data-contracts.md` |
| InferenceEngine | `ocr/inference/engine.py` |
| Pydantic Models | `apps/shared/backend_shared/models/inference.py` |

---

## Phase 1: Backend Service Extraction

### Task 1.1: Create CheckpointService

**Target**: `apps/ocr-inference-console/backend/services/checkpoint_service.py`

```python
# INTERFACE CONTRACT
class CheckpointService:
    _cache: list[Checkpoint] | None
    _last_update: datetime | None

    def __init__(self, checkpoint_root: Path, cache_ttl: float = 5.0): ...
    async def list_checkpoints(self, limit: int = 100) -> list[Checkpoint]: ...
    def get_latest(self) -> Checkpoint | None: ...
    def _discover_sync(self, limit: int) -> list[Checkpoint]: ...
```

**Changes**:
1. `[NEW]` `backend/services/__init__.py`
2. `[NEW]` `backend/services/checkpoint_service.py` - extract lines 134-211 from `main.py`
3. `[MODIFY]` `backend/main.py` - replace `_discover_checkpoints()` with service injection

**Acceptance**:
- `/api/inference/checkpoints` returns in <50ms
- TTL cache works (identical results within 5s)

---

### Task 1.2: Create InferenceService

**Target**: `apps/ocr-inference-console/backend/services/inference_service.py`

```python
# INTERFACE CONTRACT
class InferenceService:
    _engine: InferenceEngine | None
    _current_checkpoint: str | None

    def __init__(self): ...
    async def predict(
        self,
        image: np.ndarray,
        checkpoint_path: str,
        params: InferenceRequest
    ) -> InferenceResponse: ...
    def cleanup(self) -> None: ...
```

**Changes**:
1. `[NEW]` `backend/services/inference_service.py` - extract lines 276-382 from `main.py`
2. `[NEW]` `backend/services/preprocessing_service.py` - extract base64 decoding (lines 308-327)
3. `[MODIFY]` `backend/main.py` - use service injection, ~150 lines reduced

**Acceptance**:
- Inference works with new service structure
- Same checkpoint reuse (no reload if unchanged)

---

### Task 1.3: Structured Error Handling

**Target**: `apps/ocr-inference-console/backend/exceptions.py`

```python
# ERROR HIERARCHY
class OCRBackendError(Exception):
    error_code: str
    message: str
    details: dict

class CheckpointNotFoundError(OCRBackendError): ...
class ImageDecodingError(OCRBackendError): ...
class InferenceError(OCRBackendError): ...
class ModelLoadError(OCRBackendError): ...
```

**Changes**:
1. `[NEW]` `backend/exceptions.py`
2. `[NEW]` `backend/models/errors.py` - ErrorResponse Pydantic model
3. `[MODIFY]` `backend/main.py` - use structured exceptions

**Acceptance**:
- All errors return `{error_code, message, request_id}`
- HTTPException uses detail dict, not string

---

## Phase 2: Frontend Context Migration

### Task 2.1: Create InferenceContext

**Target**: `apps/ocr-inference-console/src/contexts/InferenceContext.tsx`

```typescript
// INTERFACE CONTRACT
interface InferenceState {
  checkpoints: Checkpoint[];
  loadingCheckpoints: boolean;
  selectedCheckpoint: string | null;
  inferenceOptions: InferenceOptions;
}

interface InferenceActions {
  setSelectedCheckpoint: (path: string) => void;
  updateInferenceOptions: (opts: Partial<InferenceOptions>) => void;
  refreshCheckpoints: () => Promise<void>;
}

export function InferenceProvider({ children }: Props): JSX.Element;
export function useInference(): InferenceState & InferenceActions;
```

**Changes**:
1. `[NEW]` `src/contexts/InferenceContext.tsx`
2. `[MODIFY]` `src/App.tsx` - wrap with `<InferenceProvider>`, remove state (lines 9-22)
3. `[MODIFY]` `src/components/Sidebar.tsx` - use `useInference()`, remove 14 props

**Acceptance**:
- No prop drilling for checkpoints/options
- Same functionality preserved

---

### Task 2.2: Migrate Workspace Component

**Changes**:
1. `[MODIFY]` `src/components/Workspace.tsx` - use `useInference()`, remove 13 props
2. `[MODIFY]` `src/components/TopRibbon.tsx` - use context if needed

**Acceptance**:
- Component renders correctly
- Inference flow unchanged

---

## Phase 3: Async Checkpoint Preloading

### Task 3.1: Background Model Loading

**Target**: Enhanced `CheckpointService`

```python
# ADDITIONS TO CheckpointService
async def preload_checkpoint(self, path: str) -> None: ...
async def _load_model_background(self, path: str) -> None: ...
```

**Changes**:
1. `[MODIFY]` `backend/services/checkpoint_service.py` - add preload methods
2. `[MODIFY]` `backend/main.py` lifespan - start background preload task

**Acceptance**:
- Server starts in <2s
- Background preload logs appear
- First inference doesn't block if preload completed

---

## Proposed Changes Summary

### Backend Files

| Action | File | Description |
|--------|------|-------------|
| NEW | `backend/services/__init__.py` | Package init |
| NEW | `backend/services/checkpoint_service.py` | Checkpoint discovery + caching |
| NEW | `backend/services/inference_service.py` | InferenceEngine lifecycle |
| NEW | `backend/services/preprocessing_service.py` | Image decoding/validation |
| NEW | `backend/exceptions.py` | Error hierarchy |
| NEW | `backend/models/errors.py` | ErrorResponse model |
| MODIFY | `backend/main.py` | Reduce to ~150 lines, use services |

### Frontend Files

| Action | File | Description |
|--------|------|-------------|
| NEW | `src/contexts/InferenceContext.tsx` | Centralized state |
| MODIFY | `src/App.tsx` | Wrap with provider, remove state |
| MODIFY | `src/components/Sidebar.tsx` | Use context hook |
| MODIFY | `src/components/Workspace.tsx` | Use context hook |

---

## Verification Plan

### Automated Tests

> [!NOTE]
> No existing unit tests found for OCR console backend. Manual verification required.

**Commands to verify**:
```bash
# Backend smoke test
cd apps/ocr-inference-console && python -c "from backend.services.checkpoint_service import CheckpointService; print('OK')"

# Frontend type check
cd apps/ocr-inference-console && npm run build

# Full stack test (requires running servers)
make ocr-console-backend &
make serve-ocr-console &
curl -s http://127.0.0.1:8002/api/health | jq .status
```

### Manual Verification

| Test | Steps | Expected |
|------|-------|----------|
| Checkpoint List | 1. Start backend<br>2. Open console<br>3. Check sidebar | Checkpoints load, spinner shows briefly |
| Inference Flow | 1. Upload image<br>2. Select checkpoint<br>3. Click "Run Inference" | Polygons overlay on image |
| Options Persist | 1. Change confidence threshold<br>2. Re-run inference | New threshold applied |
| Error Display | 1. Use invalid checkpoint path<br>2. Check error message | Structured error with error_code |
| Performance | 1. Time `/api/inference/checkpoints`<br>2. Repeat within 5s | <50ms, cached |

### Browser Testing

```
Task: Verify OCR console after refactoring
1. Navigate to http://127.0.0.1:5173
2. Wait for checkpoints to load in sidebar
3. Upload test image (any JPEG)
4. Verify polygon overlay appears
5. Change NMS threshold slider
6. Click "Run Inference"
7. Verify new results with changed threshold
Return: Pass/fail with screenshots
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Service extraction breaks API | Keep endpoints unchanged, only internal refactor |
| Context migration breaks rendering | Migrate one component at a time with fallback props |
| Background preload race conditions | Use asyncio.Lock for model loading |
| Type mismatches after context | Run `npm run build` after each component |

---

## Execution Order

```mermaid
graph LR
    A[Task 1.1: CheckpointService] --> B[Task 1.2: InferenceService]
    B --> C[Task 1.3: Error Handling]
    C --> D[Task 2.1: InferenceContext]
    D --> E[Task 2.2: Migrate Workspace]
    E --> F[Task 3.1: Background Loading]
```

**Dependencies**:
- Phase 2 requires Phase 1 complete (backend must work)
- Phase 3 requires Phase 1 complete (services must exist)
- Tasks within phases can be executed sequentially by single worker

---

## Worker Assignment Hints

Each phase can be executed by an autonomous worker:

| Worker | Scope | Est. Duration |
|--------|-------|---------------|
| Worker 1 | Phase 1 (Backend) | 2-3h |
| Worker 2 | Phase 2 (Frontend) | 2-3h |
| Worker 3 | Phase 3 (Async) | 1-2h |

**Worker Context Bundle**:
```yaml
required_files:
  - apps/ocr-inference-console/backend/main.py
  - apps/shared/backend_shared/models/inference.py
  - docs/artifacts/specs/shared-backend-contract.md
  - apps/ocr-inference-console/docs/data-contracts.md
```
