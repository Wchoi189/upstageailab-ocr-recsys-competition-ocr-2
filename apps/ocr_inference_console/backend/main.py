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
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure backend directory is in python path to handle imports correctly
# when running via uvicorn with various configurations or reloads
backend_dir = str(Path(__file__).parent.resolve())
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Heavy imports (cv2, numpy, torch, lightning) moved to local scopes inside functions
# Import exceptions and models
from exceptions import (
    CheckpointNotFoundError,
    ImageDecodingError,
    InferenceError,
    OCRBackendError,
    ServiceNotInitializedError,
)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models.errors import ErrorResponse
from pydantic import BaseModel

# Import services
from services.checkpoint_service import Checkpoint, CheckpointService
from services.inference_service import InferenceService
from services.preprocessing_service import PreprocessingService

# Import shared models (lightweight pure-Pydantic models)
from apps.shared.backend_shared.models.inference import (
    InferenceMetadata,
    InferenceRequest,
    InferenceResponse,
    Padding,
    TextRegion,
)

logger = logging.getLogger(__name__)

# Resolve checkpoint root relative to project root (3 levels up from backend dir)
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent.parent.parent

# Structured logging to file + console for backend diagnostics
LOG_DIR = PROJECT_ROOT / "outputs" / "logs" / "ocr-inference-console"
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

    # Also set DEBUG for ocr.core.inference logger
    inference_logger = logging.getLogger("ocr.core.inference")
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
    _checkpoint_service = CheckpointService(checkpoint_root=DEFAULT_CHECKPOINT_ROOT, cache_ttl=5.0)
    _inference_service = InferenceService()

    async def _startup_task():
        """Background task to warm up the model."""
        try:
            logger.info("⏳ Waiting for checkpoint service to initialize...")
            # Wait for checkpoint preload to complete (it's already started above)
            # We can just poll or trust the service - let's wait a moment for FS ops
            await asyncio.sleep(0.5)

            assert _checkpoint_service is not None, "Checkpoint service not initialized"
            assert _inference_service is not None, "Inference service not initialized"

            latest = await _checkpoint_service.get_latest_async()
            if latest:
                logger.info("🚀 Triggering background warm-up for: %s", latest.checkpoint_path)
                await _inference_service.warm_up(latest.checkpoint_path)
            else:
                logger.warning("⚠️ No checkpoints found. Skipping warm-up.")
        except Exception as e:
            logger.error("❌ Startup task failed: %s", e)

    # Background preload checkpoint cache (non-blocking)
    asyncio.create_task(_checkpoint_service.preload_checkpoints())

    # Fire-and-forget background warm-up task
    # We don't await this because we want the server to start immediately
    asyncio.create_task(_startup_task())

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


class ExtractionRequest(BaseModel):
    """Request model for receipt extraction endpoint.

    Attributes:
        image_base64: Base64-encoded image data
        enable_vlm: Whether to enable VLM for complex receipts
        checkpoint_path: Optional checkpoint path (uses latest if not specified)
    """

    image_base64: str
    enable_vlm: bool = True
    checkpoint_path: str | None = None


class ExtractionResponse(BaseModel):
    """Response model for receipt extraction endpoint.

    Attributes:
        detection_result: Detection results (polygons and texts)
        receipt_data: Extracted structured receipt data
        processing_time_ms: Total processing time in milliseconds
        vlm_used: Whether VLM was used for extraction
    """

    detection_result: dict
    receipt_data: dict
    processing_time_ms: float
    vlm_used: bool


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

    import asyncio
    import time

    start_time = time.perf_counter()
    loop = asyncio.get_running_loop()

    # Decode base64 image using preprocessing service (offloaded to threadpool)
    try:
        image = await loop.run_in_executor(None, PreprocessingService.decode_base64_image, request.image_base64 or "")
    except ValueError as e:
        raise ImageDecodingError(str(e))

    # Run inference using inference service
    try:
        logger.info(
            "📥 Inference request: perspective=%s, grayscale=%s, bg_norm=%s, sepia=%s, clahe=%s",
            request.enable_perspective_correction,
            request.enable_grayscale,
            request.enable_background_normalization,
            request.enable_sepia_enhancement,
            request.enable_clahe,
        )
        result = await _inference_service.predict(
            image=image,
            checkpoint_path=str(ckpt_path),
            confidence_threshold=request.confidence_threshold,
            nms_threshold=request.nms_threshold,
            enable_perspective_correction=request.enable_perspective_correction,
            perspective_display_mode=request.perspective_display_mode,
            enable_grayscale=request.enable_grayscale,
            enable_background_normalization=request.enable_background_normalization,
            enable_sepia_enhancement=request.enable_sepia_enhancement,
            enable_clahe=request.enable_clahe,
            sepia_display_mode=request.sepia_display_mode,
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

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    return InferenceResponse(
        status="success",
        regions=regions,
        meta=meta,
        preview_image_base64=result.get("preview_image_base64"),
        processing_time_ms=processing_time_ms,
        notes=[],  # Placeholder for future warnings/info
    )


@app.post(f"{API_PREFIX}/inference/extract", response_model=ExtractionResponse)
async def extract_receipt(request: ExtractionRequest):
    """Extract structured data from receipt image.

    This endpoint runs the full extraction pipeline:
    1. Text detection
    2. Text recognition
    3. Layout grouping
    4. Field extraction (rule-based + optional VLM)

    Returns structured receipt data with store info, items, and totals.
    """
    # Validate services initialized
    if _checkpoint_service is None:
        raise ServiceNotInitializedError("CheckpointService")
    if _inference_service is None:
        raise ServiceNotInitializedError("InferenceService")

    # Resolve checkpoint path
    checkpoint_path = request.checkpoint_path
    if not checkpoint_path:
        latest = _checkpoint_service.get_latest()
        if latest is None:
            raise CheckpointNotFoundError("No checkpoints available")
        checkpoint_path = latest.checkpoint_path

    # Validate checkpoint exists
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise CheckpointNotFoundError(checkpoint_path)

    import asyncio
    import base64
    import time

    import cv2
    import numpy as np

    start_time = time.perf_counter()
    loop = asyncio.get_running_loop()

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise ImageDecodingError(str(e))

    # Run extraction pipeline
    try:
        # This needs to be done in executor to avoid blocking
        def _run_extraction():
            # Ensure orchestrator has extraction pipeline enabled
            orchestrator = _inference_service.get_orchestrator()
            if orchestrator is None:
                raise ServiceNotInitializedError("Orchestrator not initialized")

            if not orchestrator._enable_extraction:
                orchestrator.enable_extraction_pipeline()

            # Run prediction with extraction enabled
            result = orchestrator.predict(
                image,
                return_preview=False,
                enable_extraction=True,
            )
            return result

        result = await loop.run_in_executor(None, _run_extraction)

        if result is None:
            raise InferenceError("Extraction pipeline returned None")

    except Exception as e:
        raise InferenceError(str(e))

    elapsed = (time.perf_counter() - start_time) * 1000

    # Check if VLM was used (heuristic: check metadata)
    receipt_data_dict = result.get("receipt_data", {})
    metadata = receipt_data_dict.get("metadata", {})
    model_version = metadata.get("model_version", "")
    vlm_used = model_version.startswith("VLM:")

    return ExtractionResponse(
        detection_result={
            "polygons": result.get("polygons"),
            "texts": result.get("recognized_texts", result.get("texts")),
        },
        receipt_data=receipt_data_dict,
        processing_time_ms=elapsed,
        vlm_used=vlm_used,
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
