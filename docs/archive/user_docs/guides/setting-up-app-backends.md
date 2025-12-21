# Setting Up Domain-Driven App Backends

**Last Updated**: December 14, 2025
**Status**: Complete
**Architecture**: Domain-Driven Separation

## Overview

Following the domain-driven separation migration, each frontend application manages its own backend. This guide explains how to:

1. **Set up a backend** for your app
2. **Use shared packages** (`apps/shared/backend_shared/`)
3. **Access model checkpoints** and process inference requests
4. **Deploy with RESTful APIs**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Vite/Next.js)                  │
│           apps/ocr-inference-console/ or                    │
│           apps/playground-console/                          │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/JSON
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  App-Specific Backend (FastAPI)              │
│     apps/<app>/backend/ (Port 8001 or 8002)                  │
├──────────────────────────────────────────────────────────────┤
│  • /api/v1/inference/*                                       │
│  • /api/v1/checkpoints/*                                     │
│  • /api/v1/health                                            │
└──────────────────────┬───────────────────────────────────────┘
                       │ Imports & Uses
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              Shared Backend Components                       │
│        apps/shared/backend_shared/                           │
├──────────────────────────────────────────────────────────────┤
│  • InferenceEngine (from ocr/inference/)                     │
│  • Pydantic Models (validation, type safety)                 │
│  • Shared utilities                                          │
└──────────────────────────────────────────────────────────────┘
                       │ Uses
                       ▼
┌──────────────────────────────────────────────────────────────┐
│           Core OCR & ML Libraries                            │
│  • ocr/ (training, inference, evaluation)                    │
│  • lightning_modules/ (PyTorch Lightning models)             │
│  • Checkpoints (outputs/experiments/train/ocr/)              │
└──────────────────────────────────────────────────────────────┘
```

---

## Step 1: Create Backend Structure

### 1.1 Directory Setup

For **OCR Inference Console**, create:

```bash
apps/ocr-inference-console/backend/
├── __init__.py
├── main.py                 # FastAPI app entry point
├── config.py              # Configuration & settings
├── routers/
│   ├── __init__.py
│   ├── inference.py       # OCR inference endpoints
│   ├── checkpoints.py     # Checkpoint discovery endpoints
│   └── health.py          # Health check endpoints
├── models/
│   ├── __init__.py
│   └── schemas.py         # Pydantic request/response models
└── utils/
    ├── __init__.py
    └── checkpoint_loader.py  # Checkpoint loading utilities
```

### 1.2 Main Entry Point (`main.py`)

```python
"""
OCR Inference Console Backend - FastAPI Server

Serves the OCR Inference Console frontend with inference and checkpoint endpoints.
Uses shared InferenceEngine from apps/shared/backend_shared/.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from .routers import inference, checkpoints, health

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
OCR_CHECKPOINT_PATH = os.getenv("OCR_CHECKPOINT_PATH")
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8002"))

# Global state
_inference_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for app startup/shutdown.
    Initializes the InferenceEngine on startup.
    """
    global _inference_engine

    # Startup
    logger.info("🚀 Starting OCR Inference Console Backend...")

    # Initialize InferenceEngine
    from apps.shared.backend_shared.inference.engine import InferenceEngine

    if not OCR_CHECKPOINT_PATH:
        logger.warning("⚠️  OCR_CHECKPOINT_PATH not set. Inference will be unavailable.")
    else:
        try:
            _inference_engine = InferenceEngine(checkpoint_path=OCR_CHECKPOINT_PATH)
            logger.info(f"✅ InferenceEngine initialized with checkpoint: {OCR_CHECKPOINT_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize InferenceEngine: {e}")
            _inference_engine = None

    yield

    # Shutdown
    logger.info("🛑 Shutting down OCR Inference Console Backend...")
    if _inference_engine:
        _inference_engine.cleanup()


# Create FastAPI app
app = FastAPI(
    title="OCR Inference Console Backend",
    description="RESTful API for OCR model inference and checkpoint management",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(checkpoints.router, prefix="/api/v1", tags=["checkpoints"])
app.include_router(inference.router, prefix="/api/v1", tags=["inference"])


def get_inference_engine():
    """Get the global InferenceEngine instance."""
    if _inference_engine is None:
        raise RuntimeError("InferenceEngine not initialized. Check OCR_CHECKPOINT_PATH.")
    return _inference_engine


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True  # Set to False in production
    )
```

---

## Step 2: Use Shared Packages

### 2.1 Import InferenceEngine

The `InferenceEngine` is the core class from `ocr/inference/` moved to `apps/shared/backend_shared/`:

```python
# In your routers/inference.py
from apps.shared.backend_shared.inference.engine import InferenceEngine

# Usage
engine = get_inference_engine()  # From main.py
predictions = engine.predict(image_input)
```

### 2.2 Use Pydantic Models

Shared models provide type safety and validation:

```python
# In your models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class InferenceRequest(BaseModel):
    """OCR inference request."""
    image: str = Field(..., description="Base64-encoded image")
    config: Optional[dict] = Field(None, description="Inference config")

class InferenceResponse(BaseModel):
    """OCR inference response."""
    success: bool
    predictions: dict
    processing_time_ms: float
```

---

## Step 3: Create API Endpoints

### 3.1 Health Check Router (`routers/health.py`)

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    """Check if backend is running."""
    return {
        "status": "ok",
        "service": "ocr-inference-console-backend",
        "version": "1.0.0"
    }
```

### 3.2 Checkpoints Router (`routers/checkpoints.py`)

```python
from fastapi import APIRouter, HTTPException
from pathlib import Path
import os

router = APIRouter()

@router.get("/checkpoints")
async def list_checkpoints():
    """List available OCR checkpoints."""
    checkpoint_dir = Path("outputs/experiments/train/ocr")

    if not checkpoint_dir.exists():
        raise HTTPException(status_code=404, detail="No checkpoints found")

    checkpoints = [
        {
            "path": str(f),
            "name": f.stem,
            "size_mb": f.stat().st_size / (1024 * 1024),
        }
        for f in checkpoint_dir.glob("*.ckpt")
    ]

    return {"checkpoints": checkpoints}

@router.get("/checkpoints/current")
async def get_current_checkpoint():
    """Get the currently loaded checkpoint."""
    checkpoint_path = os.getenv("OCR_CHECKPOINT_PATH")

    if not checkpoint_path:
        raise HTTPException(status_code=404, detail="No checkpoint loaded")

    return {
        "path": checkpoint_path,
        "name": Path(checkpoint_path).stem
    }
```

### 3.3 Inference Router (`routers/inference.py`)

```python
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
import base64
import io
from PIL import Image
import time

from ..models.schemas import InferenceRequest, InferenceResponse
from ..main import get_inference_engine

router = APIRouter()

@router.post("/inference/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Run OCR inference on an image."""

    try:
        engine = get_inference_engine()

        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))

        # Run inference
        start = time.time()
        predictions = engine.predict(image, **request.config or {})
        processing_time = (time.time() - start) * 1000  # Convert to ms

        return InferenceResponse(
            success=True,
            predictions=predictions,
            processing_time_ms=processing_time
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )

@router.post("/inference/predict-file")
async def predict_file(file: UploadFile = File(...)):
    """Run OCR inference on an uploaded file."""

    try:
        engine = get_inference_engine()

        # Read uploaded file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run inference
        start = time.time()
        predictions = engine.predict(image)
        processing_time = (time.time() - start) * 1000

        return {
            "success": True,
            "predictions": predictions,
            "processing_time_ms": processing_time,
            "filename": file.filename
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )
```

---

## Step 4: Configure and Start Backend

### 4.1 Environment Setup

```bash
# Export checkpoint path
export OCR_CHECKPOINT_PATH="outputs/experiments/train/ocr/checkpoint.ckpt"

# Export port (optional, defaults to 8002)
export BACKEND_PORT=8002

# Export host (optional, defaults to 127.0.0.1)
export BACKEND_HOST=0.0.0.0
```

### 4.2 Start with Makefile

Add target to main `Makefile`:

```makefile
ocr-console-backend:
	@bash -c 'set -euo pipefail; \
		export OCR_CHECKPOINT_PATH=$$(find outputs/experiments/train/ocr -name "*.ckpt" | head -n 1); \
		if [ -z "$$OCR_CHECKPOINT_PATH" ]; then \
			echo "Error: No checkpoint found. Set OCR_CHECKPOINT_PATH manually."; \
			exit 1; \
		fi; \
		echo "Starting OCR Console backend on http://127.0.0.1:8002"; \
		cd apps/ocr-inference-console/backend && \
		uv run uvicorn main:app --host 127.0.0.1 --port 8002 --reload; \
	'
```

### 4.3 Start Manually

```bash
# With automatic checkpoint detection
export OCR_CHECKPOINT_PATH=$(find outputs/experiments/train/ocr -name "*.ckpt" | head -n 1)
cd apps/ocr-inference-console/backend
uv run uvicorn main:app --host 127.0.0.1 --port 8002 --reload

# Visit: http://localhost:8002/docs (Swagger UI)
```

---

## Step 5: Connect Frontend to Backend

### 5.1 Frontend API Client

```typescript
// apps/ocr-inference-console/src/api/client.ts

const API_BASE = process.env.VITE_API_URL || "http://localhost:8002/api/v1";

export async function predict(imageBase64: string) {
  const response = await fetch(`${API_BASE}/inference/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image: imageBase64,
      config: {}
    })
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return response.json();
}

export async function listCheckpoints() {
  const response = await fetch(`${API_BASE}/checkpoints`);
  return response.json();
}

export async function getHealth() {
  const response = await fetch(`${API_BASE}/health`);
  return response.json();
}
```

### 5.2 Environment Configuration

```bash
# .env for frontend development
VITE_API_URL=http://localhost:8002/api/v1
```

---

## Step 6: Testing & Debugging

### 6.1 Interactive API Documentation

Once backend is running, visit:
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

### 6.2 Test Health Endpoint

```bash
curl http://localhost:8002/api/v1/health
# Response: {"status":"ok","service":"ocr-inference-console-backend","version":"1.0.0"}
```

### 6.3 Test Inference Endpoint

```bash
# With a test image
curl -X POST http://localhost:8002/api/v1/inference/predict-file \
  -F "file=@test_image.png"
```

---

## Common Issues & Solutions

### Issue 1: "InferenceEngine not initialized"

**Cause**: `OCR_CHECKPOINT_PATH` not set or checkpoint file not found

**Solution**:
```bash
# Find checkpoint
find outputs/experiments/train/ocr -name "*.ckpt" | head -n 1

# Set environment variable
export OCR_CHECKPOINT_PATH="/path/to/checkpoint.ckpt"

# Restart backend
```

### Issue 2: Backend doesn't respond / Port already in use

**Solution**:
```bash
# Kill all hanging processes
make kill-ports

# Or manually specify different port
export BACKEND_PORT=8003
```

### Issue 3: CORS errors in frontend

**Solution**: Update CORS configuration in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## What's Next?

1. **Add more routers** as needed (preprocessing, post-processing, etc.)
2. **Add request validation** with Pydantic models
3. **Add logging** with structured logs
4. **Add monitoring** with Prometheus metrics
5. **Deploy** with Docker or production ASGI server (Gunicorn)

See the archived backend at `docs/archive/archive_code/deprecated/apps-backend/` for more complex examples.

---

## References

- **Implementation Plan**: [Domain-Driven Separation](../artifacts/implementation_plans/2025-12-14_0220_implementation_plan_domain-driven-separation.md)
- **Deprecation Manifest**: [Archive](../archive/archive_code/deprecated/DEPRECATION_MANIFEST.md)
- **Shared Backend**: `apps/shared/backend_shared/`
- **FastAPI Docs**: https://fastapi.tiangolo.com/
- **Pydantic Docs**: https://docs.pydantic.dev/
