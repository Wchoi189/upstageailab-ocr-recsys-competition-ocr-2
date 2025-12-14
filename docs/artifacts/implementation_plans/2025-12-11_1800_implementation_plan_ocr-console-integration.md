---
type: "implementation_plan"
category: "development"
status: "active"
version: "1.0"
tags: ['implementation', 'plan', 'ocr', 'integration']
title: "Implementation Plan - OCR PIPELINE INTEGRATION"
date: "2025-12-11 18:00 (KST)"
branch: "feature/agentqms-dashboard-integration"
---

# Implementation Plan - OCR PIPELINE INTEGRATION

**Goal**: Integrate the existing OCR pipeline (`ocr/`) into the `ocr-inference-console` application, enabling users to run inference on images and view polygon results.

## References

**Source Documents**:
- **Requirements**: [Draft Prompt](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/draft-prompt.md)
- **Analysis**: [Assessment](file:///home/vscode/.gemini/antigravity/brain/d7be2dea-5484-4d6d-b291-675ca2dddf8d/draft_prompt_assessment.md)

**Technical Contracts**:
- **Data Contracts**: `docs/pipeline/data_contracts.md` (Note: user mentioned `docs/architecture`, but verified path is `docs/pipeline/data_contracts.md`)
- **Validation Models**: `ocr/validation/models.py`

## User Review Required

> [!IMPORTANT]
> **Architecture Decision**: The draft prompt suggested creating `apps/ocr-inference-console/adapters/ocr_bridge.py`. Since `ocr-inference-console` is a client-side React app, it cannot run Python code directly. I will instead implement the API bridge in `apps/backend/services/ocr_bridge.py` (or extending existing backend services) and expose it via REST API.

## Proposed Changes

### Documentation
**Component**: `apps/ocr-inference-console/docs`

#### [NEW] Documentation Structure
- Create directory structure: `integration/`, `data/`, `development/`, `meta/`
- Add `README.md` with navigation
- Add `meta/agent-instructions.md` (AI system prompts)
- Add `data_contracts.md` (schemas)

### Data Validator
**Component**: `apps/ocr-inference-console/scripts`

#### [NEW] [validate_data.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/scripts/validate_data.py)
- CLI script to validate dataset existence and JSON schema
- **Dependencies**: Reuses `ocr.validation.models` (Pydantic models)
- **Output**: JSON validation report

### Backend API (OCR Bridge)
**Component**: `apps/backend`

#### [NEW] [services/ocr_bridge.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/backend/services/ocr_bridge.py)
- FastAPI router/service
- **Endpoints**:
  - `POST /predict`: Accepts image file, returns polygons
  - `GET /health`: Checks pipeline status
- **Integration**: Imports `ocr.lightning_modules.ocr_pl` to load checkpoints and run inference

### Frontend (React UI)
**Component**: `apps/ocr-inference-console`

#### [NEW] [src/api/ocrClient.ts](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/src/api/ocrClient.ts)
- TypeScript API client for the backend bridge

#### [NEW] [src/components/ImageUploader.tsx](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/src/components/ImageUploader.tsx)
- Component to select/upload images

#### [NEW] [src/components/PolygonOverlay.tsx](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/src/components/PolygonOverlay.tsx)
- Canvas/SVG overlay to draw polygons on images

#### [MODIFY] [src/App.tsx](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/src/App.tsx)
- Integrate new components into main layout

## Verification Plan

### Automated Tests
- **Validator Test**:
  ```bash
  python apps/ocr-inference-console/scripts/validate_data.py --path data/datasets --sample 5
  ```
  *Expect: Exit code 0, JSON output with "status": "ok"*

- **API Smoke Test**:
  ```bash
  # Start backend (background)
  uv run uvicorn apps.backend.main:app --port 8000 &
  # Query health
  curl http://localhost:8000/health
  # Query prediction (mock or real)
  curl -X POST -F "file=@data/datasets/images_val_canonical/sample.jpg" http://localhost:8000/predict
  ```

- **Frontend Tests**:
  ```bash
  cd apps/ocr-inference-console
  npm run test
  ```

- **Existing Tests**:
  - Run `uv run pytest tests/integration/test_ocr_pipeline_integration.py` to ensure core pipeline is stable before integration.

### Manual Verification
1. Start backend: `uv run uvicorn apps.backend.main:app`
2. Start frontend: `cd apps/ocr-inference-console && npm run dev`
3. Open `http://localhost:5173`
4. Upload an image from `data/datasets/images_val_canonical/`
5. Verify that polygons are drawn correctly over the text regions.
