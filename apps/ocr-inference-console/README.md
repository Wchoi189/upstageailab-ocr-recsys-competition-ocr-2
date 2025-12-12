# OCR Inference Console

**Type**: Modern Web Application
**Tech Stack**: Vite + React + TypeScript + TailwindCSS
**Status**: 🟡 70% Complete (Core features working, needs polish)

---

## Purpose

Lightweight, inference-focused web console for running OCR text detection on images. Designed for quick testing and demonstration of OCR models.

---

## Features

### Implemented ✅
- Image upload and inference
- Polygon overlay rendering on detected text regions
- Demo mode with sample data
- Checkpoint selection
- Real-time prediction visualization
- Data contract validation

### In Progress 🟡
- Batch image processing
- Confidence threshold adjustment
- Export results

### Planned ⚪
- Preprocessing options
- Model comparison

---

## Tech Stack

- **Frontend**: Vite + React 18 + TypeScript
- **Styling**: TailwindCSS
- **State**: React Hooks
- **Backend**: FastAPI (`apps/backend/services/ocr_bridge.py`)

---

## How to Run

### 1. Start Backend

```bash
# From project root
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Set checkpoint path (or let it auto-detect latest)
export OCR_CHECKPOINT_PATH="outputs/experiments/train/ocr/.../epoch-14.ckpt"

# Start backend
python -m uvicorn apps.backend.main:app --port 8000
```

### 2. Start Frontend

```bash
# From project root
cd apps/ocr-inference-console

# Install dependencies (first time only)
npm install

# Start dev server
npm run dev

# Open http://localhost:5173
```

---

## API Integration

This console depends on the FastAPI backend at `apps/backend/`.

**Endpoints Used**:
- `GET /ocr/health` - Health check
- `POST /ocr/predict` - Run OCR inference
- `GET /ocr/checkpoints` - List available checkpoints

**Data Contract**: See [docs/data-contracts.md](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/docs/data-contracts.md)

---

## Architecture

```
apps/ocr-inference-console/
├── src/
│   ├── components/
│   │   ├── Workspace.tsx          # Main inference UI
│   │   ├── PolygonOverlay.tsx      # Renders detected polygons
│   │   ├── Sidebar.tsx            # Checkpoint selection
│   │   └── TopRibbon.tsx          # Navigation
│   ├── api/
│   │   └── ocrClient.ts           # Backend API client
│   └── App.tsx                    # Root component
├── public/
│   ├── demo.jpg                   # Demo image
│   └── demo.json                  # Demo predictions
└── docs/
    └── data-contracts.md          # API contracts
```

---

## Development Guidelines

### Code Quality
- TypeScript strict mode enabled
- ESLint + Prettier configured
- Component-based architecture

### Testing Your Changes
1. Test with demo mode (click "Demo" button)
2. Test with real image upload
3. Verify polygon overlays render correctly
4. Check browser console for errors

---

## Backend Dependency

This app uses `apps/backend/services/ocr_bridge.py`, which wraps the proven `ui.utils.inference.InferenceEngine`.

**Key Point**: The backend automatically handles:
- Model loading (lazy, on first request)
- Image preprocessing
- Coordinate transformation
- Polygon extraction

See [OCR Bridge Refactoring Walkthrough](file:///home/vscode/.gemini/antigravity/brain/e233fabb-0950-4377-903d-e30dbc71cd13/walkthrough.md) for technical details.

---

## Related Documentation

- [System Overview](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/architecture/00_system_overview.md)
- [Data Contracts](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/docs/data-contracts.md)
- [Backend README](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/backend/README.md)
- [Project Roadmap](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/roadmap.md)

---

## Status: 70% Complete

**What Works**:
- ✅ Core inference flow
- ✅ Polygon rendering
- ✅ Demo mode
- ✅ Checkpoint selection

**What Needs Work**:
- 🟡 Batch processing
- 🟡 Advanced controls (thresholds, etc.)
- 🟡 Export/download results
- 🟡 Error handling improvements

---

**For Questions**: See [System Overview](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/architecture/00_system_overview.md) or contact project maintainers.
