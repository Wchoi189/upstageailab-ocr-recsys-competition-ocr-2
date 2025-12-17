---
type: guide
component: playground_spa
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# High-Performance Playground

**Purpose**: React+Vite SPA for OCR experimentation; Albumentations-style interface with Web Workers (<100ms latency) and FastAPI backend.

---

## Features

| Feature | Route | Purpose |
|---------|-------|---------|
| **Preprocessing Studio** | `/preprocessing` | Real-time image enhancement with before/after preview |
| **Inference Studio** | `/inference` | Interactive model testing with polygon overlays |
| **Comparison Studio** | `/comparison` | Side-by-side model comparison, A/B testing, gallery |
| **Command Builder** | `/commands` | Visual interface for training/testing commands |
| **Telemetry Dashboard** | `/metrics` | Real-time performance metrics, worker statistics |

---

## Quick Start

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| **Node.js** | 18+ | Frontend runtime |
| **Python** | 3.10+ | Backend runtime |
| **UV** | Latest | Python package manager |

### Installation

```bash
# Frontend dependencies
cd frontend && npm install

# Backend dependencies (if needed)
cd .. && uv sync
```

### Running

**Option 1 - Unified Launcher (Recommended)**:
```bash
python run_spa.py
```
Opens FastAPI (`:8000`) + Vite (`:5173`) + browser automatically

**Option 2 - Manual Start**:
```bash
# Terminal 1: Backend
uvicorn services.playground_api.app:app --reload --port 8000

# Terminal 2: Frontend
cd frontend && npm run dev
```

---

## Studio Guides

### Preprocessing Studio (`/preprocessing`)

| Feature | Implementation | Performance |
|---------|----------------|-------------|
| **Before/after canvas** | WebGL comparison | Real-time |
| **Parameter controls** | Sliders, toggles | <100ms response |
| **Auto Contrast** | Web Worker | <100ms |
| **Gaussian Blur** | Web Worker (kernel: 3-15px) | <100ms |
| **Resize** | Web Worker (128-2048px) | <100ms |
| **Background Removal** | ONNX.js (client) / FastAPI (backend) | <400ms (client), <800ms (backend) |

### Inference Studio (`/inference`)

| Feature | Implementation | Notes |
|---------|----------------|-------|
| **Single/batch modes** | Tab interface | Checkpoint selector |
| **Polygon overlays** | Canvas rendering | Interactive visualization |
| **Confidence threshold** | Slider (0-1) | Real-time filtering |
| **NMS threshold** | Slider (0-1) | Post-processing |

### Comparison Studio (`/comparison`)

| Feature | Implementation | Notes |
|---------|----------------|-------|
| **A/B testing** | Side-by-side canvas | Diff visualization |
| **Gallery view** | Masonry grid | Lazy loading |
| **Metrics table** | CSV upload | Precision, recall, F1 |

### Command Builder (`/commands`)

| Feature | Implementation | Notes |
|---------|----------------|-------|
| **Training forms** | Schema-driven | `/api/commands/build` |
| **Validation** | Real-time | Error messages |
| **CLI preview** | Syntax-highlighted | Copy/download |

---

## Architecture

### Client-Side Processing

| Component | Technology | Latency Target |
|-----------|------------|----------------|
| **Web Workers** | TypeScript + Comlink | <100ms |
| **ONNX.js** | WASM SIMD | <400ms (rembg) |
| **Canvas** | WebGL | 60fps |

### Backend Services

| Endpoint | Purpose | Technology |
|----------|---------|------------|
| `/api/commands/build` | Generate CLI commands | FastAPI |
| `/api/inference/preview` | Heavy compute fallback | FastAPI + PyTorch |
| `/api/pipelines/preview` | Pipeline preview | FastAPI |
| `/api/checkpoints` | Checkpoint catalog | FastAPI |

---

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| **Slider response** | <100ms | Lightweight transforms |
| **Client rembg** | <400ms | ONNX.js, <2048px images |
| **Backend fallback** | <800ms | >2048px or throttled CPU |
| **Canvas refresh** | 60fps | WebGL rendering |
| **Worker queue** | <5 tasks | During slider spam |

---

## Dependencies

| Dependency | Purpose |
|-----------|---------|
| **React** | UI framework |
| **Vite** | Build tool, dev server |
| **Comlink** | Worker RPC |
| **ONNX.js** | Client-side ML inference |
| **FastAPI** | Backend API |
| **PyTorch** | Heavy compute |

---

## Constraints

- SharedArrayBuffer requires secure context (HTTPS or localhost)
- ONNX.js rembg model: ~3MB download
- Worker pool: max `min(cores - 1, 6)`
- Backend fallback: automatic for >2048px images

---

## Backward Compatibility

**Status**: New system (v1.0)

**Breaking Changes**: N/A (replaces Streamlit apps)

**Migration**: Streamlit apps remain available during transition; feature parity tracked in [parity.md](parity.md)

---

## References

- [Design System](design-system.md)
- [Worker Blueprint](worker-blueprint.md)
- [Testing Observability](testing-observability.md)
- [Migration Roadmap](migration-roadmap.md)
- [Parity Matrix](parity.md)
