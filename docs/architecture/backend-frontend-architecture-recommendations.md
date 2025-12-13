# Backend/Frontend Architecture Recommendations

**Date**: 2025-01-20
**Status**: Recommendations for refactoring
**Context**: Current architecture has unclear separation of concerns between `apps/backend/`, `apps/frontend/`, and shared packages.

---

## Current Architecture Issues

### Problem 1: Unclear App Boundaries

**Current State:**
- `apps/backend/` serves both `playground-console` and `ocr-inference-console`
- `apps/frontend/` exists but purpose is unclear (appears to be a separate frontend app)
- Shared packages (`@upstage/console-shared`) exist but usage is inconsistent
- Both consoles have their own API clients that sometimes bypass shared packages

**Issues:**
1. **No clear ownership**: Which app owns which backend endpoints?
2. **Code duplication**: Both consoles implement similar API clients
3. **Tight coupling**: Backend services are mixed (`ocr_bridge.py` vs `playground_api/`)
4. **Unclear `apps/frontend/` purpose**: Is it a third app? Legacy? Future?

### Problem 2: Shared Package Inconsistency

**Current State:**
- `@upstage/console-shared` exists but:
  - `ocr-inference-console` bypasses it with direct `fetch` calls
  - `playground-console` may or may not use it consistently
  - No clear contract for when to use shared vs direct calls

**Issues:**
1. **Inconsistent patterns**: Some code uses shared, some uses direct fetch
2. **Maintenance burden**: Changes require updates in multiple places
3. **No versioning strategy**: Shared package changes affect all consumers

### Problem 3: Backend Service Organization

**Current State:**
```
apps/backend/
├── main.py                    # Entry point, includes both routers
├── services/
│   ├── ocr_bridge.py         # OCR Inference Console endpoints
│   └── playground_api/       # Playground Console endpoints
│       ├── app.py
│       └── routers/
```

**Issues:**
1. **Mixed concerns**: `ocr_bridge.py` is a service, `playground_api/` is a full API
2. **Inconsistent patterns**: Different routing/organization styles
3. **No clear API versioning**: Both use different path prefixes (`/ocr/*` vs `/api/*`)

---

## Recommended Architecture

### Option A: Domain-Driven Separation (Recommended)

**Principle**: Each frontend app has its own dedicated backend service.

```
apps/
├── playground-console/          # Next.js app
│   ├── frontend/                # React components
│   └── backend/                 # Dedicated FastAPI service
│       ├── main.py
│       └── routers/
│           ├── inference.py
│           └── command_builder.py
│
├── ocr-inference-console/       # Vite+React app
│   ├── frontend/                # React components
│   └── backend/                 # Dedicated FastAPI service
│       ├── main.py
│       └── routers/
│           └── inference.py
│
└── shared/                      # Shared packages
    ├── console-shared/          # TypeScript shared utilities
    │   ├── api/                 # API client interfaces
    │   ├── types/               # Shared TypeScript types
    │   └── utils/               # Shared utilities
    │
    └── backend-shared/          # Python shared backend code
        ├── inference/           # Shared inference logic
        │   └── engine.py        # InferenceEngine (moved from ui/utils)
        └── models/              # Shared data models
```

**Benefits:**
- ✅ Clear ownership: Each app owns its backend
- ✅ Independent deployment: Apps can be deployed separately
- ✅ Clear boundaries: No confusion about which service serves which app
- ✅ Easier testing: Test each app's backend independently
- ✅ Shared code in dedicated packages: Clear reuse strategy

**Migration Path:**
1. Create `apps/playground-console/backend/` from `apps/backend/services/playground_api/`
2. Create `apps/ocr-inference-console/backend/` from `apps/backend/services/ocr_bridge.py`
3. Move shared inference logic to `apps/shared/backend-shared/inference/`
4. Update `apps/backend/main.py` to be a simple router that delegates (temporary)
5. Deprecate `apps/backend/` after migration

---

### Option B: Unified Backend with Clear Modules

**Principle**: Single backend with clear module boundaries and versioned APIs.

```
apps/
├── backend/                     # Unified FastAPI backend
│   ├── main.py                 # Entry point, CORS, routing
│   ├── api/
│   │   ├── v1/
│   │   │   ├── playground/     # Playground Console endpoints
│   │   │   │   ├── inference.py
│   │   │   │   └── command_builder.py
│   │   │   └── ocr/            # OCR Inference Console endpoints
│   │   │       └── inference.py
│   │   └── shared/             # Shared API utilities
│   │       └── models.py
│   │
│   └── services/                # Business logic
│       ├── inference/         # Shared inference service
│       │   └── engine.py
│       └── checkpoints/        # Checkpoint management
│
├── playground-console/          # Next.js app (frontend only)
├── ocr-inference-console/      # Vite+React app (frontend only)
│
└── shared/
    └── console-shared/          # TypeScript shared package
        ├── api/
        │   ├── playground.ts   # Playground API client
        │   └── ocr.ts          # OCR API client
        └── types/
```

**Benefits:**
- ✅ Single deployment: One backend service
- ✅ Shared infrastructure: CORS, auth, logging configured once
- ✅ Versioned APIs: Clear API versioning strategy
- ✅ Consistent patterns: All endpoints follow same structure

**Drawbacks:**
- ❌ Tight coupling: Changes affect both apps
- ❌ Deployment complexity: Can't deploy apps independently
- ❌ Scaling: Can't scale apps separately

---

### Option C: Microservices (Overkill for Current Scale)

**Not recommended** unless you have:
- Multiple teams working independently
- Different scaling requirements per app
- Different deployment schedules
- Need for independent service discovery

---

## Recommendation: Option A (Domain-Driven Separation)

### Rationale

1. **Clear Ownership**: Each app team owns their full stack
2. **Independent Evolution**: Apps can evolve at different paces
3. **Easier Onboarding**: New developers understand "this app = this backend"
4. **Better Testing**: Test each app's backend in isolation
5. **Future-Proof**: Easy to extract services later if needed

### Implementation Plan

#### Phase 1: Create Shared Packages (Week 1)

```bash
# Create shared backend package
mkdir -p apps/shared/backend-shared/inference
# Move InferenceEngine from ui/utils/inference/engine.py
mv ui/utils/inference/engine.py apps/shared/backend-shared/inference/

# Update console-shared package structure
apps/shared/console-shared/
├── src/
│   ├── api/
│   │   ├── playground-client.ts
│   │   └── ocr-client.ts
│   ├── types/
│   │   ├── inference.ts
│   │   └── checkpoints.ts
│   └── utils/
```

#### Phase 2: Extract Playground Backend (Week 2)

```bash
# Create playground-console backend
mkdir -p apps/playground-console/backend
# Move from apps/backend/services/playground_api/
cp -r apps/backend/services/playground_api/* apps/playground-console/backend/

# Update imports to use shared backend package
# Update playground-console to use its own backend
```

#### Phase 3: Extract OCR Backend (Week 3)

```bash
# Create ocr-inference-console backend
mkdir -p apps/ocr-inference-console/backend
# Extract from apps/backend/services/ocr_bridge.py
# Create proper FastAPI structure

# Update ocr-inference-console to use its own backend
```

#### Phase 4: Deprecate Unified Backend (Week 4)

```bash
# Update apps/backend/main.py to be a simple router
# Add deprecation warnings
# Update documentation
# Remove after 1 month
```

---

## Shared Package Strategy

### Backend Shared (`apps/shared/backend-shared/`)

**Purpose**: Shared Python code used by multiple backend services.

**Contents:**
- `inference/engine.py` - InferenceEngine (core OCR logic)
- `models/` - Shared Pydantic models
- `utils/` - Shared utilities (path utils, etc.)

**Usage:**
```python
# In playground-console/backend/routers/inference.py
from apps.shared.backend_shared.inference.engine import InferenceEngine
```

### Frontend Shared (`apps/shared/console-shared/`)

**Purpose**: Shared TypeScript code used by multiple frontend apps.

**Contents:**
- `api/` - API client implementations
- `types/` - Shared TypeScript interfaces
- `utils/` - Shared utilities

**Usage:**
```typescript
// In playground-console
import { PlaygroundClient } from '@upstage/console-shared/api/playground-client';
```

**Versioning Strategy:**
- Use semantic versioning
- Publish to npm (private registry or local)
- Apps pin specific versions
- Breaking changes require major version bump

---

## API Design Principles

### 1. Consistent Endpoint Patterns

**Current (Inconsistent):**
- `/ocr/predict` (OCR Inference Console)
- `/api/inference/preview` (Playground Console)

**Recommended:**
- `/api/v1/ocr/inference` (OCR Inference Console)
- `/api/v1/playground/inference` (Playground Console)

Or with domain-driven separation:
- Playground: `http://localhost:8001/api/v1/inference`
- OCR: `http://localhost:8002/api/v1/inference`

### 2. Shared Data Contracts

**Location**: `apps/shared/backend-shared/models/`

```python
# apps/shared/backend-shared/models/inference.py
from pydantic import BaseModel

class InferenceRequest(BaseModel):
    image_base64: str
    checkpoint_path: str | None = None

class InferenceResponse(BaseModel):
    status: str
    regions: list[TextRegion]
    meta: InferenceMetadata
```

**Usage**: Both backends import and use these models.

### 3. API Client Interfaces

**Location**: `apps/shared/console-shared/src/api/`

```typescript
// apps/shared/console-shared/src/api/inference-client.ts
export interface InferenceClient {
  predict(request: InferenceRequest): Promise<InferenceResponse>;
  listCheckpoints(): Promise<Checkpoint[]>;
}

// apps/shared/console-shared/src/api/playground-client.ts
export class PlaygroundClient implements InferenceClient {
  // Implementation
}

// apps/shared/console-shared/src/api/ocr-client.ts
export class OCRClient implements InferenceClient {
  // Implementation
}
```

---

## Migration Checklist

### Immediate Actions (This Week)

- [ ] Document current architecture (this document)
- [ ] Create `apps/shared/backend-shared/` structure
- [ ] Move `InferenceEngine` to shared package
- [ ] Update imports across codebase

### Short Term (Next Month)

- [ ] Extract playground-console backend
- [ ] Extract ocr-inference-console backend
- [ ] Update shared TypeScript package structure
- [ ] Update API clients to use shared interfaces

### Long Term (Next Quarter)

- [ ] Deprecate `apps/backend/` unified backend
- [ ] Remove `apps/frontend/` if unused
- [ ] Establish API versioning strategy
- [ ] Set up CI/CD for independent deployments

---

## Questions to Resolve

1. **What is `apps/frontend/`?**
   - Is it a third app?
   - Legacy code?
   - Should it be removed or documented?

2. **Deployment Strategy**
   - Do apps need independent deployment?
   - Or is unified deployment acceptable?

3. **Shared Package Publishing**
   - Local npm registry?
   - Git submodules?
   - Monorepo tooling (Turborepo, Nx)?

4. **API Versioning**
   - When to introduce v2?
   - How to handle breaking changes?

---

## References

- [Current System Overview](./00_system_overview.md)
- [API Decoupling Strategy](./api-decoupling.md)
- [Inference Data Contracts](../pipeline/inference-data-contracts.md)
