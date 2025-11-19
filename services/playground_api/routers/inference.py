"""Inference API router providing checkpoint discovery and job stubs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import yaml
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..utils.paths import PROJECT_ROOT
from ocr.utils.experiment_name import resolve_experiment_name

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

    NOTE: This is a stub implementation. Actual inference wiring pending.
    """
    # Placeholder: Return mock data for UI development
    mock_regions = [
        TextRegion(
            polygon=[[100, 100], [300, 100], [300, 150], [100, 150]],
            text="Sample Text 1",
            confidence=0.95,
        ),
        TextRegion(
            polygon=[[100, 200], [400, 200], [400, 250], [100, 250]],
            text="Sample Text 2",
            confidence=0.87,
        ),
    ]

    return InferencePreviewResponse(
        status="success",
        regions=mock_regions,
        processing_time_ms=125.5,
        notes=["Stub implementation - actual inference not wired yet"],
    )


