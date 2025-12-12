# OCR Backend (FastAPI)

**Type**: REST API Backend
**Tech Stack**: Python + FastAPI + PyTorch
**Status**: ✅ Active Development

---

## Purpose

FastAPI backend serving both modern Next.js consoles (Playground Console and OCR Inference Console). Provides OCR inference, checkpoint management, and playground API endpoints.

---

## Endpoints

### OCR Inference Console (`/ocr/*`)

- `GET /ocr/health` - Health check, model status
- `POST /ocr/predict` - Run OCR inference on uploaded image
- `GET /ocr/checkpoints` - List available model checkpoints

### Playground Console (`/api/*`)

- `GET /api/inference/modes` - List inference modes
- `GET /api/inference/checkpoints` - Checkpoint catalog
- `POST /api/inference/preview` - Run inference with preview
- Additional command builder and preprocessing endpoints

### Documentation

- `GET /docs` - Swagger/OpenAPI documentation
- `GET /` - API status and info

---

## How to Run

### Prerequisites

```bash
# Ensure you're in project root
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Install Python dependencies (if not already)
pip install -r requirements.txt
```

### Start Server

```bash
# Option 1: Auto-detect latest checkpoint
python -m uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000

# Option 2: Specify checkpoint explicitly
export OCR_CHECKPOINT_PATH="outputs/experiments/train/ocr/pan_resnet18_add_polygons_canonical/20241019_0033_00/checkpoints/epoch-14_step-001545.ckpt"
python -m uvicorn apps.backend.main:app --host 0.0.0.0 --port 8000

# Server starts at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

---

## Architecture

```
apps/backend/
├── main.py                        # FastAPI app + CORS
├── services/
│   ├── ocr_bridge.py             # OCR inference (wraps InferenceEngine)
│   └── playground_api/           # Playground Console API
│       ├── app.py
│       └── routers/
│           ├── inference.py      # Inference endpoints
│           └── command_builder.py
```

---

## Key Components

### 1. OCR Bridge (`services/ocr_bridge.py`)

Wraps the proven `ui.utils.inference.InferenceEngine` to provide inference for the OCR Inference Console.

**Features**:
- Lazy model loading (fast server startup)
- Automatic checkpoint detection
- Image preprocessing
- Coordinate transformation
- Polygon extraction

**Data Contract**: Returns array format for frontend compatibility:
```json
{
  "filename": "image.jpg",
  "predictions": [
    {"points": [[x,y], ...], "confidence": 0.95}
  ]
}
```

See [OCR Bridge Walkthrough](file:///home/vscode/.gemini/antigravity/brain/e233fabb-0950-4377-903d-e30dbc71cd13/walkthrough.md) for details.

### 2. Playground API (`services/playground_api/`)

Full API for the Playground Console, including:
- Checkpoint catalog with metadata
- Inference with preview images
- Command builder endpoints
- Configuration management

**Data Contract**: Returns string format with metadata:
```json
{
  "status": "success",
  "regions": [...],
  "meta": {...},
  "preview_image_base64": "..."
}
```

---

## Dependencies

### Shared Logic

This backend uses `ui.utils.inference.InferenceEngine` - the battle-tested inference engine from Legacy Streamlit apps.

**Why?**: Eliminates code duplication, ensures correct coordinate transformation.

**Documented Dependency**: See [System Overview - Shared Logic](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/architecture/00_system_overview.md#shared-logic-architecture)

### Environment Variables

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `OCR_CHECKPOINT_PATH` | No | Auto-detect | Path to model checkpoint |
| `PORT` | No | 8000 | Server port |

See [Environment Variables Documentation](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/architecture/environment-variables.md) for complete list.

---

## Development Guidelines

### Adding New Endpoints

1. Create router in `services/`
2. Include router in `main.py`
3. Test with Swagger docs at `/docs`
4. Update this README

### Testing

```bash
# Health check
curl http://localhost:8000/ocr/health | jq '.'

# Inference
curl -X POST http://localhost:8000/ocr/predict \
  -F "file=@test_image.jpg" | jq '.'

# Interactive testing
# Open http://localhost:8000/docs
```

---

## Performance

**Startup Time**: <5 seconds (lazy loading)
**First Request**: 5-30 seconds (model loading)
**Subsequent Requests**: <2 seconds (model cached)

---

## Related Documentation

- [System Overview](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/architecture/00_system_overview.md)
- [OCR Inference Console](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/ocr-inference-console/README.md)
- [Playground Console](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/apps/playground-console/README.md)
- [API Decoupling Architecture](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/docs/architecture/api-decoupling.md)
