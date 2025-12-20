# OCR Inference Console

**Type**: Modern Web Application
**Tech Stack**: Vite + React + TypeScript + TailwindCSS
**Status**: 🟢 Production Ready (Refactored Dec 21, 2025)

> **AI Agents**: See [`.ai-instructions/INDEX.yaml`](.ai-instructions/INDEX.yaml) for machine-parseable documentation

---

## Purpose

Lightweight, inference-focused web console for running OCR text detection on images. Designed for quick testing and demonstration of OCR models.

---

## Quick Start

### Start Backend + Frontend

```bash
# From project root
make ocr-console-stack
```

**Backend**: `http://localhost:8002` (port 8002)
**Frontend**: `http://localhost:5173` (port 5173)

### Start Backend Only

```bash
make ocr-console-backend
```

**Health Check**: `http://localhost:8002/api/health`
**API Docs**: `http://localhost:8002/docs`

### Start Frontend Only

```bash
make ocr-console-dev
```

---

## Architecture

### Backend (Port 8002)

**Module**: `apps/ocr-inference-console/backend/main.py`
**Framework**: FastAPI + Pydantic
**Services**:
- `CheckpointService` - Checkpoint discovery with TTL caching
- `InferenceService` - InferenceEngine lifecycle management
- `PreprocessingService` - Image decoding and validation

**API Endpoints**:
- `GET /api/health` - Health check
- `GET /api/inference/checkpoints` - List available checkpoints
- `POST /api/inference/preview` - Run OCR inference

**Error Handling**: Structured exceptions with `OCRBackendError` hierarchy

### Frontend (Port 5173)

**Framework**: React 18 + TypeScript + Vite
**State Management**: InferenceContext (React Context API)
**Components**:
- `Workspace` - Main inference UI
- `PolygonOverlay` - Renders detected text regions
- `Sidebar` - Checkpoint selection and inference controls
- `TopRibbon` - Navigation and upload

**Features**:
- Real-time inference with polygon overlay
- Demo mode with sample data
- Adjustable confidence/NMS thresholds
- Perspective correction toggle
- Grayscale and background normalization

---

## Features

### Implemented ✅
- Image upload and inference
- Polygon overlay rendering on detected text regions
- Demo mode with sample data
- Checkpoint selection with auto-discovery
- Real-time prediction visualization
- Confidence/NMS threshold adjustment
- Perspective correction toggle
- Preprocessing options (grayscale, background normalization)
- Data contract validation
- Structured error handling

### Planned ⚪
- Batch image processing
- Export results to JSON/CSV
- Model comparison view

---

## API Integration

**Backend Contract**: See `apps/ocr-inference-console/backend/` for service layer

**Shared Models** (from `apps/shared/backend_shared/models/inference.py`):
- `InferenceRequest` - Input schema
- `InferenceResponse` - Output schema with regions and metadata
- `TextRegion` - Detected text polygon with confidence
- `InferenceMetadata` - Coordinate system and padding info

**Data Contract**: See [`.ai-instructions/contracts/pydantic-models.yaml`](.ai-instructions/contracts/pydantic-models.yaml)

---

## Development

### Code Quality
- TypeScript strict mode enabled
- ESLint + Prettier configured
- Component-based architecture
- Backend service layer pattern
- React Context for state management

### Testing
1. Test with demo mode (click "Demo" button)
2. Test with real image upload
3. Verify polygon overlays render correctly
4. Check browser console for errors
5. Verify backend health endpoint

---

## Recent Changes

**Dec 21, 2025 - Backend/Frontend Refactoring**:
- Extracted service layer (CheckpointService, InferenceService, PreprocessingService)
- Implemented structured error handling with `OCRBackendError` hierarchy
- Migrated frontend to InferenceContext (eliminated 41 props from prop drilling)
- Added async checkpoint cache preloading on startup
- Reduced main.py from 400 to ~250 lines

**Related Documentation**:
- Implementation Plan: `docs/artifacts/implementation_plans/2025-12-21_0210_implementation_plan_ocr-console-refactor.md`
- Changelog Entry: See `CHANGELOG.md` (2025-12-21 03:30)

---

## Troubleshooting

### Backend Won't Start
```bash
# Check if checkpoint exists
find outputs/experiments/train/ocr -name "*.ckpt" | head -5

# Kill any hanging processes
make kill-ports
```

### Port Already in Use
```bash
# Check what's using port 8002
lsof -i:8002

# Force kill
make kill-ports
```

### Frontend Can't Connect to Backend
1. Verify backend is running: `curl http://localhost:8002/api/health`
2. Check browser console for CORS errors
3. Ensure ports match (backend: 8002, frontend: 5173)

---

## Related Documentation

- [AI Documentation Index](.ai-instructions/INDEX.yaml) - Machine-parseable contracts for AI agents
- [Shared Backend Contract](../../docs/artifacts/specs/shared-backend-contract.md)
- [System Architecture](../../docs/architecture/system-architecture.md)
- [Legacy Docs](DEPRECATED/docs/) - Archived human-oriented documentation

---

**Status**: Production Ready (Refactored 2025-12-21)
