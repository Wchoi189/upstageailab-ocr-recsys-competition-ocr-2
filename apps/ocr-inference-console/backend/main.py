"""OCR Inference Console Backend - FastAPI Server

Port 8002 | Domain: OCR Text Detection Inference
Serves OCR console frontend with checkpoint discovery and inference endpoints.
Uses shared InferenceEngine from apps.shared.backend_shared.

See: docs/guides/setting-up-app-backends.md
"""

from __future__ import annotations

import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import shared backend components
from apps.shared.backend_shared.inference import InferenceEngine
from apps.shared.backend_shared.models.inference import (
    InferenceMetadata,
    InferenceRequest,
    InferenceResponse,
    Padding,
    TextRegion,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

API_PREFIX = "/api"

# Resolve checkpoint root relative to project root (3 levels up from backend dir)
BACKEND_DIR = Path(__file__).parent
PROJECT_ROOT = BACKEND_DIR.parent.parent.parent
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs/experiments/train/ocr"

# Global inference engine (lazy loaded)
_inference_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup/shutdown."""
    global _inference_engine

    logger.info("🚀 Starting OCR Inference Console Backend (Port 8002)")

    # Initialize engine (model loads on first inference request)
    _inference_engine = InferenceEngine()
    logger.info("✅ InferenceEngine initialized (lazy loading enabled)")

    yield

    logger.info("🛑 Shutting down OCR Inference Console Backend")


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


class Checkpoint(BaseModel):
    """Checkpoint metadata for UI selection."""

    checkpoint_path: str
    display_name: str
    size_mb: float
    modified_at: str
    epoch: int | None = None
    global_step: int | None = None
    precision: float | None = None
    recall: float | None = None
    hmean: float | None = None


def _discover_checkpoints(limit: int = 100) -> list[Checkpoint]:
    """Discover available checkpoints using pregenerated metadata YAML files.

    Fast loading strategy:
    - Searches for .ckpt.metadata.yaml files instead of .ckpt files
    - Parses YAML metadata without loading checkpoint state dict
    - Provides near-instant checkpoint discovery

    Args:
        limit: Maximum number of checkpoints to return

    Returns:
        List of Checkpoint objects sorted by modification time (newest first)
    """
    if not DEFAULT_CHECKPOINT_ROOT.exists():
        logger.warning("Checkpoint root missing: %s", DEFAULT_CHECKPOINT_ROOT)
        return []

    # Find all pregenerated metadata files
    metadata_files = sorted(
        DEFAULT_CHECKPOINT_ROOT.rglob("*.ckpt.metadata.yaml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    results: list[Checkpoint] = []
    for meta_path in metadata_files[:limit]:
        try:
            # Parse YAML metadata (fast - no state dict loading)
            with open(meta_path, "r") as f:
                meta = yaml.safe_load(f)

            # Reconstruct checkpoint path from metadata file path
            ckpt_path = meta_path.with_suffix("").with_suffix("")  # Remove .metadata.yaml

            # Get file stats
            stat = meta_path.stat()
            ckpt_stat = ckpt_path.stat() if ckpt_path.exists() else stat

            display_name = str(ckpt_path.relative_to(DEFAULT_CHECKPOINT_ROOT))

            results.append(
                Checkpoint(
                    checkpoint_path=str(ckpt_path),
                    display_name=display_name,
                    size_mb=round(ckpt_stat.st_size / (1024 * 1024), 2) if ckpt_path.exists() else 0.0,
                    modified_at=meta.get("created_at", datetime.fromtimestamp(stat.st_mtime).isoformat()),
                    epoch=meta.get("training", {}).get("epoch"),
                    global_step=meta.get("training", {}).get("global_step"),
                    precision=meta.get("metrics", {}).get("precision"),
                    recall=meta.get("metrics", {}).get("recall"),
                    hmean=meta.get("metrics", {}).get("hmean"),
                )
            )
        except Exception as e:
            logger.warning("Failed to parse metadata file %s: %s", meta_path, e)
            continue

    return results


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
        "engine_loaded": _inference_engine is not None,
    }


@app.get(f"{API_PREFIX}/inference/checkpoints", response_model=list[Checkpoint])
async def list_checkpoints(limit: int = 100):
    """List available OCR checkpoints."""
    checkpoints = _discover_checkpoints(limit=limit)
    return checkpoints


@app.post(f"{API_PREFIX}/inference/preview", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run OCR inference on an image.

    Accepts base64-encoded images and returns detected text regions with
    coordinate metadata for frontend overlay rendering.
    """
    if _inference_engine is None:
        raise HTTPException(status_code=503, detail="InferenceEngine not initialized")

    # Resolve checkpoint path
    checkpoint_path = request.checkpoint_path
    if not checkpoint_path:
        # Use latest checkpoint if not specified
        checkpoints = _discover_checkpoints(limit=1)
        if not checkpoints:
            raise HTTPException(status_code=404, detail="No checkpoints available")
        checkpoint_path = checkpoints[0].checkpoint_path

    # Validate checkpoint exists
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise HTTPException(status_code=400, detail=f"Checkpoint not found: {checkpoint_path}")

    # Decode base64 image
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 required")

    try:
        # Handle data URL prefix
        image_b64 = request.image_base64
        if "base64," in image_b64:
            image_b64 = image_b64.split("base64,", 1)[1]

        # Decode to numpy array
        image_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding failed: {str(e)}")

    # Load model if needed
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    if not _inference_engine.load_model(str(ckpt_path)):
        raise HTTPException(status_code=500, detail="Failed to load model checkpoint")

    # Run inference
    try:
        result = _inference_engine.predict_array(
            image_array=image,
            binarization_thresh=request.confidence_threshold,
            box_thresh=request.nms_threshold,
            return_preview=True,  # Get 640x640 preview with metadata
        )

        if result is None:
            raise RuntimeError("Inference returned None")

    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

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
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("BACKEND_HOST", "127.0.0.1"),
        port=int(os.getenv("BACKEND_PORT", "8002")),
        reload=True,
    )
