"""OCR Inference Console Backend - FastAPI Server

Port 8002 | Domain: OCR Text Detection Inference
Serves OCR console frontend with checkpoint discovery and inference endpoints.
Uses shared InferenceEngine from apps.shared.backend_shared.

See: docs/guides/setting-up-app-backends.md
"""

from __future__ import annotations

# Lightweight imports only at top level for instant startup
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

# Heavy imports (cv2, numpy, torch, lightning) moved to local scopes inside functions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import shared models (lightweight pure-Pydantic models)
from apps.shared.backend_shared.models.inference import (
    InferenceMetadata,
    InferenceRequest,
    InferenceResponse,
    Padding,
    TextRegion,
)

# Import services
from .services.checkpoint_service import Checkpoint, CheckpointService
from .services.inference_service import InferenceService
from .services.preprocessing_service import PreprocessingService

# Import exceptions and models
from .exceptions import (
    CheckpointNotFoundError,
    ImageDecodingError,
    InferenceError,
    OCRBackendError,
    ServiceNotInitializedError,
)
from .models.errors import ErrorResponse

logger = logging.getLogger(__name__)

# Resolve checkpoint root relative to project root (3 levels up from backend dir)
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent.parent.parent

# Structured logging to file + console for backend diagnostics
LOG_DIR = PROJECT_ROOT / "logs" / "ocr-inference-console"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "backend.log"

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Capture DEBUG in file
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)  # Set root level to DEBUG
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Also set DEBUG for ocr.inference logger
    inference_logger = logging.getLogger("ocr.inference")
    inference_logger.setLevel(logging.DEBUG)

API_PREFIX = "/api"

DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs/experiments/train/ocr"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup/shutdown."""
    import asyncio
    global _checkpoint_service, _inference_service

    logger.info("🚀 Starting OCR Inference Console Backend (Port 8002)")

    # Initialize services
    _checkpoint_service = CheckpointService(
        checkpoint_root=DEFAULT_CHECKPOINT_ROOT,
        cache_ttl=5.0
    )
    _inference_service = InferenceService()

    # Background preload checkpoint cache (non-blocking)
    asyncio.create_task(_checkpoint_service.preload_checkpoints())

    # Note: InferenceEngine initialization is now lazy-loaded on first inference request
    # to ensure the server starts responding to health/discovery requests immediately.

    yield

    logger.info("🛑 Shutting down OCR Inference Console Backend")

    # Clean up service resources to prevent memory/semaphore leaks
    if _inference_service is not None:
        try:
            _inference_service.cleanup()
        except Exception as e:
            logger.error(f"Error during inference service cleanup: {e}")

    _checkpoint_service = None
    _inference_service = None
    logger.info("✅ Shutdown complete")


app = FastAPI(
    title="OCR Inference Console Backend",
    description="RESTful API for OCR text detection inference and checkpoint management",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers for structured errors
@app.exception_handler(OCRBackendError)
async def ocr_backend_error_handler(request, exc: OCRBackendError):
    """Handle structured OCR backend errors."""
    from fastapi.responses import JSONResponse

    status_code = 500  # Default to internal server error

    # Map specific error codes to HTTP status codes
    status_map = {
        "CHECKPOINT_NOT_FOUND": 404,
        "IMAGE_DECODING_ERROR": 400,
        "INFERENCE_ERROR": 500,
        "MODEL_LOAD_ERROR": 500,
        "SERVICE_NOT_INITIALIZED": 500,
    }

    status_code = status_map.get(exc.error_code, 500)

    error_response = ErrorResponse(
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(),
    )


# Initialize services
_checkpoint_service: CheckpointService | None = None
_inference_service: InferenceService | None = None


def _parse_inference_result(result: dict) -> list[TextRegion]:
    """Parse InferenceEngine result into TextRegion list.

    Args:
        result: Dict with 'polygons', 'texts', 'confidences' keys

    Returns:
        List of TextRegion objects
    """
    polygons_str = result.get("polygons", "")
    texts = result.get("texts", [])
    confidences = result.get("confidences", [])

    if not polygons_str:
        return []

    polygon_groups = polygons_str.split("|")
    regions = []

    for idx, polygon_str in enumerate(polygon_groups):
        coords = polygon_str.strip().split()
        if len(coords) < 6:  # Need at least 3 points
            continue

        try:
            coord_floats = [float(c) for c in coords]
            polygon = [[coord_floats[i], coord_floats[i + 1]] for i in range(0, len(coord_floats), 2)]
        except (ValueError, IndexError):
            logger.warning(f"Failed to parse polygon: {polygon_str}")
            continue

        text = texts[idx] if idx < len(texts) else None
        confidence = confidences[idx] if idx < len(confidences) else 0.0

        regions.append(
            TextRegion(
                polygon=polygon,
                text=text,
                confidence=confidence,
            )
        )

    return regions


@app.get(f"{API_PREFIX}/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "checkpoint_root": str(DEFAULT_CHECKPOINT_ROOT),
        "engine_loaded": _inference_service is not None and _inference_service._engine is not None,
    }


@app.get(f"{API_PREFIX}/inference/checkpoints", response_model=list[Checkpoint])
async def list_checkpoints(limit: int = 100):
    """List available OCR checkpoints."""
    if _checkpoint_service is None:
        raise ServiceNotInitializedError("CheckpointService")
    checkpoints = await _checkpoint_service.list_checkpoints(limit=limit)
    return checkpoints


@app.post(f"{API_PREFIX}/inference/preview", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run OCR inference on an image.

    Accepts base64-encoded images and returns detected text regions with
    coordinate metadata for frontend overlay rendering.
    """
    # Validate services initialized
    if _checkpoint_service is None:
        raise ServiceNotInitializedError("CheckpointService")
    if _inference_service is None:
        raise ServiceNotInitializedError("InferenceService")

    # Resolve checkpoint path
    checkpoint_path = request.checkpoint_path
    if not checkpoint_path:
        # Use latest checkpoint if not specified
        latest = _checkpoint_service.get_latest()
        if latest is None:
            raise CheckpointNotFoundError("No checkpoints available")
        checkpoint_path = latest.checkpoint_path

    # Validate checkpoint exists
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise CheckpointNotFoundError(checkpoint_path)

    # Decode base64 image using preprocessing service
    try:
        image = PreprocessingService.decode_base64_image(request.image_base64 or "")
    except ValueError as e:
        raise ImageDecodingError(str(e))

    # Run inference using inference service
    try:
        result = await _inference_service.predict(
            image=image,
            checkpoint_path=str(ckpt_path),
            confidence_threshold=request.confidence_threshold,
            nms_threshold=request.nms_threshold,
            enable_perspective_correction=request.enable_perspective_correction,
            perspective_display_mode=request.perspective_display_mode,
            enable_grayscale=request.enable_grayscale,
            enable_background_normalization=request.enable_background_normalization,
        )
    except RuntimeError as e:
        raise InferenceError(str(e))

    # Parse to response model
    regions = _parse_inference_result(result)

    # Extract metadata
    meta = None
    meta_dict = result.get("meta")
    if meta_dict:
        try:
            meta = InferenceMetadata(
                original_size=tuple(meta_dict["original_size"]),
                processed_size=tuple(meta_dict["processed_size"]),
                padding=Padding(**meta_dict["padding"]),
                padding_position=meta_dict.get("padding_position", "top_left"),
                content_area=tuple(meta_dict["content_area"]),
                scale=float(meta_dict["scale"]),
                coordinate_system=meta_dict.get("coordinate_system", "pixel"),
            )
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Failed to parse metadata: {e}")

    return InferenceResponse(
        status="success",
        regions=regions,
        meta=meta,
        preview_image_base64=result.get("preview_image_base64"),
    )


if __name__ == "__main__":
    import multiprocessing

    import uvicorn

    # Set multiprocessing start method to prevent semaphore leaks
    # This must be called before any multiprocessing/threading operations
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, ignore
        pass

    uvicorn.run(
        "main:app",
        host=os.getenv("BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("BACKEND_PORT", "8002")),
        reload=True,
    )
