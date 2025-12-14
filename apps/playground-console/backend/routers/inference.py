"""Inference router for Playground Console Backend

Handles OCR text detection inference with checkpoint management.
Adapted from OCR Inference Console backend for playground experimentation.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from apps.shared.backend_shared.inference import InferenceEngine
from apps.shared.backend_shared.models.inference import (
    InferenceMetadata,
    InferenceRequest,
    InferenceResponse,
    Padding,
    TextRegion,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Global inference engine reference (injected by main.py)
_inference_engine: InferenceEngine | None = None


def set_inference_engine(engine: InferenceEngine) -> None:
    """Set the global inference engine instance."""
    global _inference_engine
    _inference_engine = engine


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


@router.post("/preview", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """Run OCR inference on an image.

    Accepts base64-encoded images and returns detected text regions with
    coordinate metadata for frontend overlay rendering.
    """
    if _inference_engine is None:
        raise HTTPException(status_code=503, detail="InferenceEngine not initialized")

    # Import checkpoint discovery here to avoid circular dependency
    from .checkpoints import discover_checkpoints

    # Resolve checkpoint path
    checkpoint_path = request.checkpoint_path
    if not checkpoint_path:
        # Use latest checkpoint if not specified
        checkpoints = discover_checkpoints(limit=1)
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
