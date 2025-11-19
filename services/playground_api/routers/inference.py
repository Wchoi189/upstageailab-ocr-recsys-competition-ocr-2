"""Inference API router providing checkpoint discovery and job stubs."""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import cv2
import numpy as np
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..utils.paths import PROJECT_ROOT
from ocr.utils.experiment_name import resolve_experiment_name

# Import inference engine
try:
    from ui.utils.inference import InferenceEngine
    INFERENCE_AVAILABLE = True
except ImportError:
    INFERENCE_AVAILABLE = False

LOGGER = logging.getLogger(__name__)

router = APIRouter()

MODES_CONFIG = PROJECT_ROOT / "configs" / "ui" / "modes" / "inference.yaml"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"


class InferenceModeSummary(BaseModel):
    """Metadata describing available inference modes."""

    id: str
    description: str | None = None
    supports_batch: bool = False
    supports_background_removal: bool = False
    notes: list[str] = Field(default_factory=list)


class CheckpointSummary(BaseModel):
    """Lightweight checkpoint metadata for UI selection."""

    display_name: str
    checkpoint_path: str
    modified_at: datetime
    size_mb: float
    exp_name: str | None = None
    architecture: str | None = None
    backbone: str | None = None


@dataclass(slots=True)
class _ModeDescriptor:
    mode_id: str
    description: str | None
    supports_batch: bool
    supports_background_removal: bool
    notes: list[str]


@lru_cache(maxsize=1)
def _load_mode_descriptors() -> list[_ModeDescriptor]:
    if not MODES_CONFIG.exists():
        raise FileNotFoundError(f"Inference mode config missing at {MODES_CONFIG}")

    with open(MODES_CONFIG, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    description = config.get("description")
    supports_batch = bool(config.get("upload", {}).get("batch_mode"))
    supports_background_removal = bool(config.get("preprocessing", {}).get("background_removal", {}).get("enabled", False))

    notes: list[str] = []
    if supports_background_removal:
        notes.append("rembg-enabled preprocessing can run client-side or fall back to backend.")
    if supports_batch:
        notes.append("Batch uploads supported via directory scanning.")

    return [
        _ModeDescriptor(
            mode_id="single",
            description=description,
            supports_batch=False,
            supports_background_removal=supports_background_removal,
            notes=notes,
        ),
        _ModeDescriptor(
            mode_id="batch",
            description="Filesystem batch processing",
            supports_batch=True,
            supports_background_removal=supports_background_removal,
            notes=["Requires mounting dataset paths inside API worker."],
        ),
    ]


def _discover_checkpoints(limit: int = 50) -> list[CheckpointSummary]:
    """Scan outputs directory for checkpoint files."""
    if not OUTPUTS_ROOT.exists():
        return []

    checkpoints: list[CheckpointSummary] = []
    ckpt_paths = sorted(OUTPUTS_ROOT.rglob("*.ckpt"), reverse=True)

    for path in ckpt_paths[:limit]:
        stat = path.stat()
        rel = path.relative_to(PROJECT_ROOT)
        display_name = path.stem
        exp_name = resolve_experiment_name(path)
        checkpoints.append(
            CheckpointSummary(
                display_name=display_name,
                checkpoint_path=str(rel),
                modified_at=datetime.fromtimestamp(stat.st_mtime),
                size_mb=round(stat.st_size / (1024 * 1024), 2),
                exp_name=exp_name,
            )
        )
    return checkpoints


@router.get("/modes", response_model=list[InferenceModeSummary])
def list_modes() -> list[InferenceModeSummary]:
    """Return inference mode summaries for the frontend."""
    return [
        InferenceModeSummary(
            id=descriptor.mode_id,
            description=descriptor.description,
            supports_batch=descriptor.supports_batch,
            supports_background_removal=descriptor.supports_background_removal,
            notes=descriptor.notes,
        )
        for descriptor in _load_mode_descriptors()
    ]


@router.get("/checkpoints", response_model=list[CheckpointSummary])
def list_checkpoints(limit: int = 50) -> list[CheckpointSummary]:
    """Discover available checkpoints for use in the UI."""
    return _discover_checkpoints(limit=limit)


class InferencePreviewRequest(BaseModel):
    """Request body for single-image inference preview."""

    checkpoint_path: str
    image_base64: str | None = None
    image_path: str | None = None
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    nms_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class TextRegion(BaseModel):
    """Detected text region with polygon coordinates."""

    polygon: list[list[float]]  # [[x1, y1], [x2, y2], ...]
    text: str | None = None
    confidence: float


class InferencePreviewResponse(BaseModel):
    """Response from inference preview."""

    status: str
    regions: list[TextRegion]
    processing_time_ms: float
    notes: list[str] = Field(default_factory=list)


@router.post("/preview", response_model=InferencePreviewResponse)
def run_inference_preview(request: InferencePreviewRequest) -> InferencePreviewResponse:
    """Run single-image inference and return detected text regions.

    Supports both base64-encoded images and file paths. Performs real OCR inference
    using the loaded checkpoint model.
    """
    start_time = time.time()
    notes = []

    # Check if inference engine is available
    if not INFERENCE_AVAILABLE:
        error_msg = "Inference engine not available. OCR modules may not be installed."
        LOGGER.error(error_msg)
        raise HTTPException(status_code=503, detail=error_msg)

    # Load image from base64 or file path
    try:
        image_array = _load_image(request.image_base64, request.image_path)
    except ValueError as e:
        LOGGER.error(f"Image loading failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    # Resolve checkpoint path (handle both absolute and relative paths)
    checkpoint_path = request.checkpoint_path
    if not checkpoint_path.startswith("/"):
        checkpoint_path = str(PROJECT_ROOT / checkpoint_path)

    # Initialize inference engine and load model
    LOGGER.info(f"Loading checkpoint: {checkpoint_path}")
    engine = InferenceEngine()

    try:
        if not engine.load_model(checkpoint_path, config_path=None):
            error_msg = f"Failed to load model from checkpoint: {request.checkpoint_path}"
            LOGGER.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        LOGGER.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # Run inference with provided hyperparameters
    LOGGER.info("Running inference with thresholds: confidence=%f, nms=%f",
                request.confidence_threshold, request.nms_threshold)

    try:
        result = engine.predict_array(
            image_array=image_array,
            binarization_thresh=request.confidence_threshold,
            box_thresh=request.nms_threshold,
            max_candidates=None,  # Use model defaults
            min_detection_size=None,  # Use model defaults
        )
    except Exception as e:
        error_msg = f"Inference execution failed: {str(e)}"
        LOGGER.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    if result is None:
        error_msg = "Inference returned no results"
        LOGGER.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # Parse inference results and convert to API response format
    regions = _parse_inference_result(result)

    processing_time_ms = (time.time() - start_time) * 1000
    LOGGER.info(f"Inference completed in {processing_time_ms:.2f}ms, found {len(regions)} regions")

    return InferencePreviewResponse(
        status="success",
        regions=regions,
        processing_time_ms=processing_time_ms,
        notes=notes,
    )


def _load_image(image_base64: str | None, image_path: str | None) -> np.ndarray:
    """Load image from base64 string or file path.

    Args:
        image_base64: Base64-encoded image data (with or without data URL prefix)
        image_path: Path to image file (absolute or relative to PROJECT_ROOT)

    Returns:
        Image as numpy array in BGR format (OpenCV standard)

    Raises:
        ValueError: If both or neither image sources are provided, or if loading fails
    """
    if image_base64 and image_path:
        raise ValueError("Provide either image_base64 or image_path, not both")

    if not image_base64 and not image_path:
        raise ValueError("Must provide either image_base64 or image_path")

    if image_base64:
        # Handle data URL prefix (e.g., "data:image/png;base64,...")
        if "base64," in image_base64:
            image_base64 = image_base64.split("base64,", 1)[1]

        try:
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode base64 image data")

            return image
        except Exception as e:
            raise ValueError(f"Failed to process base64 image: {str(e)}")

    if image_path:
        # Resolve path (handle relative paths)
        path = image_path if image_path.startswith("/") else str(PROJECT_ROOT / image_path)

        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Failed to load image from path: {image_path}")
            return image
        except Exception as e:
            raise ValueError(f"Failed to load image from path: {str(e)}")

    raise ValueError("Unexpected error in image loading logic")


def _parse_inference_result(result: dict) -> list[TextRegion]:
    """Parse inference engine result into API response format.

    Args:
        result: Dictionary with 'polygons', 'texts', 'confidences' keys
                polygons: space-separated coordinate string (x1 y1 x2 y2 ... | x1 y1 ...)
                texts: list of text strings (may be None/empty)
                confidences: list of confidence scores

    Returns:
        List of TextRegion objects
    """
    polygons_str = result.get("polygons", "")
    texts = result.get("texts", [])
    confidences = result.get("confidences", [])

    if not polygons_str:
        return []

    # Parse polygons: space-separated coordinates, regions separated by "|"
    polygon_groups = polygons_str.split("|")
    regions = []

    for idx, polygon_str in enumerate(polygon_groups):
        coords = polygon_str.strip().split()
        if len(coords) < 6:  # Need at least 3 points (x1 y1 x2 y2 x3 y3)
            continue

        # Convert to list of [x, y] pairs
        try:
            coord_floats = [float(c) for c in coords]
            polygon = [[coord_floats[i], coord_floats[i+1]]
                      for i in range(0, len(coord_floats), 2)]
        except (ValueError, IndexError):
            LOGGER.warning(f"Failed to parse polygon coordinates: {polygon_str}")
            continue

        # Get corresponding text and confidence
        text = texts[idx] if idx < len(texts) else None
        confidence = confidences[idx] if idx < len(confidences) else 0.0

        regions.append(TextRegion(
            polygon=polygon,
            text=text,
            confidence=confidence,
        ))

    return regions


