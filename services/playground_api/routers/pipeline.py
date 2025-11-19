"""Pipeline preview + fallback endpoints."""

from __future__ import annotations

import base64
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..utils.paths import PROJECT_ROOT

# Import background removal
try:
    from ocr.datasets.preprocessing.background_removal import BackgroundRemoval
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

router = APIRouter()

# Global background removal instance (lazy loaded)
_bg_removal: BackgroundRemoval | None = None

# In-memory job tracker (simple implementation)
# TODO: Consider using a proper database or job queue for production
_job_statuses: dict[str, dict[str, Any]] = {}


def _get_background_removal() -> BackgroundRemoval:
    """Get or create the global BackgroundRemoval instance.

    Lazy initialization to avoid loading the model until actually needed.
    """
    global _bg_removal
    if _bg_removal is None:
        if not REMBG_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Background removal not available. rembg package not installed."
            )
        LOGGER.info("Initializing BackgroundRemoval (loading rembg model...)")
        _bg_removal = BackgroundRemoval(
            model="u2net",
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
            p=1.0,
        )
    return _bg_removal


class PipelinePreviewRequest(BaseModel):
    """Request body for client-side preview tasks."""

    pipeline_id: str = Field(description="Identifier for the logical pipeline (e.g., preprocessing, inference, comparison).")
    checkpoint_path: str | None = Field(
        default=None,
        description="Optional checkpoint reference for inference-style previews.",
    )
    image_base64: str | None = Field(
        default=None,
        description="Base64-encoded image payload supplied by the SPA.",
    )
    image_path: str | None = Field(
        default=None,
        description="Relative path to a dataset image (train/val) when the file already exists on disk.",
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Pipeline parameters / slider states.")


class PipelinePreviewResponse(BaseModel):
    status: str
    job_id: str
    routed_backend: str
    cache_key: str
    notes: list[str] = Field(default_factory=list)


class PipelineFallbackRequest(BaseModel):
    """Request body for forcing a backend fallback (e.g., heavy rembg)."""

    pipeline_id: str
    image_path: str
    params: dict[str, Any] = Field(default_factory=dict)


class PipelineFallbackResponse(BaseModel):
    status: str
    routed_backend: str
    result_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class PipelineJobStatus(BaseModel):
    """Job status response model."""

    job_id: str
    pipeline_id: str
    status: str  # pending, processing, completed, failed
    routed_backend: str
    created_at: datetime
    updated_at: datetime
    result_path: str | None = None
    error: str | None = None
    notes: list[str] = Field(default_factory=list)


def _resolve_image_bytes(image_base64: str | None, image_path: str | None) -> bytes:
    if image_base64:
        try:
            return base64.b64decode(image_base64)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Failed to decode base64 payload: {exc}") from exc

    if image_path:
        resolved = (PROJECT_ROOT / image_path).resolve()
        if not resolved.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        return resolved.read_bytes()

    raise HTTPException(status_code=400, detail="Provide either image_base64 or image_path.")


def _build_cache_key(pipeline_id: str, params: dict[str, Any], image_bytes: bytes) -> str:
    import hashlib
    import json

    param_blob = json.dumps(params, sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(image_bytes + param_blob, usedforsecurity=False).hexdigest()  # noqa: S324
    return f"{pipeline_id}:{digest[:12]}"


@router.post("/preview", response_model=PipelinePreviewResponse)
def queue_preview(request: PipelinePreviewRequest) -> PipelinePreviewResponse:
    """Validate preview payloads and acknowledge client-side execution."""
    image_bytes = _resolve_image_bytes(request.image_base64, request.image_path)
    cache_key = _build_cache_key(request.pipeline_id, request.params, image_bytes)
    backend = "client-workers"
    notes = ["Client should use cache_key to reuse previously computed previews."]
    if request.params.get("background_removal") == "server":
        backend = "server-rembg"
        notes.append("Parameter requested immediate backend fallback.")

    job_id = f"{request.pipeline_id}-{cache_key[-6:]}"
    now = datetime.utcnow()

    # Track job status
    _job_statuses[job_id] = {
        "job_id": job_id,
        "pipeline_id": request.pipeline_id,
        "status": "pending",
        "routed_backend": backend,
        "created_at": now,
        "updated_at": now,
        "result_path": None,
        "error": None,
        "notes": notes,
    }

    return PipelinePreviewResponse(
        status="accepted",
        job_id=job_id,
        routed_backend=backend,
        cache_key=cache_key,
        notes=notes,
    )


@router.post("/fallback", response_model=PipelineFallbackResponse)
def queue_fallback(request: PipelineFallbackRequest) -> PipelineFallbackResponse:
    """Execute server-side background removal.

    This endpoint handles background removal for cases where:
    - Client-side ONNX.js processing is unavailable
    - Image is too large for client-side processing
    - User explicitly requests server-side processing
    """
    # Resolve image path
    if request.image_path.startswith("/"):
        image_file = request.image_path
    else:
        image_file = str((PROJECT_ROOT / request.image_path).resolve())

    # Check file exists
    if not Path(image_file).exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")

    # Load image
    LOGGER.info(f"Loading image from: {image_file}")
    image = cv2.imread(image_file)
    if image is None:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {request.image_path}")

    # Apply background removal
    try:
        LOGGER.info("Applying background removal...")
        bg_removal = _get_background_removal()
        result = bg_removal.apply(image)
    except Exception as e:
        LOGGER.error(f"Background removal failed: {e}")
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")

    # Create output directory
    output_dir = PROJECT_ROOT / "outputs" / "playground" / request.pipeline_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save result
    input_path = Path(image_file)
    output_filename = f"{input_path.stem}_rembg.png"
    output_path = output_dir / output_filename

    LOGGER.info(f"Saving result to: {output_path}")
    cv2.imwrite(str(output_path), result)

    result_path_rel = str(output_path.relative_to(PROJECT_ROOT))
    job_id = f"{request.pipeline_id}-{Path(image_file).stem[:6]}"
    now = datetime.utcnow()

    # Update job status (if job was tracked)
    if job_id in _job_statuses:
        _job_statuses[job_id].update({
            "status": "completed",
            "updated_at": now,
            "result_path": result_path_rel,
        })
    else:
        # Create new job status entry
        _job_statuses[job_id] = {
            "job_id": job_id,
            "pipeline_id": request.pipeline_id,
            "status": "completed",
            "routed_backend": "server-rembg",
            "created_at": now,
            "updated_at": now,
            "result_path": result_path_rel,
            "error": None,
            "notes": [
                "Background removal completed successfully",
                "Model: u2net with alpha matting enabled",
                f"Output saved to: {output_filename}",
            ],
        }

    return PipelineFallbackResponse(
        status="completed",
        routed_backend="server-rembg",
        result_path=result_path_rel,
        notes=[
            "Background removal completed successfully",
            "Model: u2net with alpha matting enabled",
            f"Output saved to: {output_filename}",
        ],
    )


@router.get("/status/{job_id}", response_model=PipelineJobStatus)
def get_job_status(job_id: str) -> PipelineJobStatus:
    """Get status of a pipeline job.

    Args:
        job_id: Unique job identifier

    Returns:
        Job status information

    Raises:
        HTTPException: If job not found (404)
    """
    if job_id not in _job_statuses:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job_data = _job_statuses[job_id]
    return PipelineJobStatus(**job_data)
