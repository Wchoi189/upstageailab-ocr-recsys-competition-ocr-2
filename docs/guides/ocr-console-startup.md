# OCR Inference Console Startup Guide

This guide explains how to start the OCR Inference Console using the three available Makefile targets.

## Quick Start

### Option 1: Frontend Only (No Backend)
```bash
make serve-ocr-console
```
- Starts the Vite development server on `http://localhost:5173`
- No backend processing available
- Use when you only want to view the UI or don't have a checkpoint

### Option 2: Backend Only
```bash
make ocr-console-backend
```
- Starts the FastAPI backend server on `http://127.0.0.1:8002`
- Requires a checkpoint in `outputs/experiments/train/ocr/`
- Auto-detects the checkpoint and fails if none is found
- Swagger UI available at `http://127.0.0.1:8002/docs`
- Use when testing the API independently from the frontend

### Option 3: Full Stack (Backend + Frontend)
```bash
make ocr-console-stack
```
- Starts both backend (port 8002) and frontend (port 5173)
- Backend runs in the background, frontend in foreground
- Press `Ctrl+C` to stop both
- Use this for full application development

## What Gets Auto-Detected

All three targets automatically find the latest OCR checkpoint:

```bash
find outputs/experiments/train/ocr -name "*.ckpt" | head -n 1
```

If no checkpoint is found:
- **serve-ocr-console**: Proceeds with frontend-only mode
- **ocr-console-backend**: Fails with an error (checkpoint required)
- **ocr-console-stack**: Starts frontend-only, logs a warning about missing backend

## Manual Startup (Alternative to Makefile)

If you prefer to run directly without the Makefile:

### Start Frontend
```bash
cd apps/ocr-inference-console
npm install  # if needed
npm run dev -- --host 0.0.0.0 --port 5173
```

### Start Backend
```bash
export OCR_CHECKPOINT_PATH="/path/to/checkpoint.ckpt"
cd apps/ocr-inference-console/backend
uv run uvicorn main:app --host 127.0.0.1 --port 8002 --reload
```

## Environment Variables

You can override checkpoint detection by setting:
```bash
export OCR_CHECKPOINT_PATH="/absolute/path/to/model.ckpt"
make ocr-console-backend
```

## API Documentation

Once the backend is running, browse to:
- **Swagger UI**: `http://127.0.0.1:8002/docs`
- **ReDoc**: `http://127.0.0.1:8002/redoc`
- **OpenAPI Schema**: `http://127.0.0.1:8002/openapi.json`

## Troubleshooting

### Port Already in Use
```bash
make kill-ports  # Force kill processes on 3000, 5173, 8000, 8002
```

### Backend Fails to Start
Check if checkpoint path is valid:
```bash
find outputs/experiments/train/ocr -name "*.ckpt"
```

If empty, train a model or set `OCR_CHECKPOINT_PATH` manually.

### Frontend Can't Connect to Backend
Ensure:
1. Backend is running on port 8002
2. CORS is properly configured in backend main.py:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```
3. Frontend API URL is correctly configured (usually `http://127.0.0.1:8002`)

## Backend Implementation

For details on implementing the backend, see:
- [Setting Up App Backends](./setting-up-app-backends.md)
- [Archived deprecated backend examples](../archive/archive_code/deprecated/apps-backend/)

The backend should:
- Import `InferenceEngine` from `apps/shared/backend_shared/inference/`
- Define endpoints for health check, checkpoint info, and inference
- Use Pydantic models for request/response validation
- Handle CORS for frontend communication

## Related Targets

- `make playground-console-dev` - Start Playground Console backend on port 8001
- `make help` - See all available Makefile targets
