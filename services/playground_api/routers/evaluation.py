"""Evaluation service endpoints supporting comparison + gallery flows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..utils.paths import PROJECT_ROOT

router = APIRouter()


class ComparisonPreset(BaseModel):
    """Describes a reusable comparison preset."""

    id: str
    label: str
    required_inputs: list[str] = Field(default_factory=list)
    description: str | None = None


class ComparisonRequest(BaseModel):
    """Request body for comparison runs."""

    preset_id: str
    model_a_path: str | None = None
    model_b_path: str | None = None
    ground_truth_path: str | None = None
    image_dir: str | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)


class ComparisonResponse(BaseModel):
    """Placeholder response for comparison jobs."""

    status: str
    message: str
    next_steps: list[str] = Field(default_factory=list)


def _load_gallery_root() -> Path:
    gallery_dir = PROJECT_ROOT / "data" / "datasets" / "images" / "val"
    return gallery_dir if gallery_dir.exists() else PROJECT_ROOT


@router.get("/presets", response_model=list[ComparisonPreset])
def list_presets() -> list[ComparisonPreset]:
    """Return hard-coded presets reflecting the legacy Evaluation Viewer tabs."""
    return [
        ComparisonPreset(
            id="single_run",
            label="Single Run Analysis",
            required_inputs=["model_a_path"],
            description="Metrics + gallery for a single submission file.",
        ),
        ComparisonPreset(
            id="model_comparison",
            label="Model A/B Comparison",
            required_inputs=["model_a_path", "model_b_path"],
            description="Diff metrics and side-by-side previews.",
        ),
        ComparisonPreset(
            id="image_gallery",
            label="Image Gallery",
            required_inputs=["image_dir"],
            description="Static gallery view using dataset samples.",
        ),
    ]


@router.post("/compare", response_model=ComparisonResponse)
def queue_comparison(request: ComparisonRequest) -> ComparisonResponse:
    """Validate comparison inputs and enqueue a placeholder job."""
    required = next((preset.required_inputs for preset in list_presets() if preset.id == request.preset_id), None)
    if required is None:
        raise HTTPException(status_code=404, detail=f"Unknown preset_id '{request.preset_id}'")

    missing = [field for field in required if getattr(request, field) in (None, "")]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields for preset '{request.preset_id}': {missing}")

    next_steps = [
        "Hook this endpoint into the worker-driven evaluation pipeline.",
        "Emit WebSocket events so the SPA can stream metrics.",
    ]
    return ComparisonResponse(status="accepted", message="Comparison job acknowledged", next_steps=next_steps)


@router.get("/gallery-root")
def gallery_root() -> dict[str, str]:
    """Expose gallery root directory so the frontend knows where to pick initial samples."""
    root = _load_gallery_root()
    return {"gallery_root": str(root.relative_to(PROJECT_ROOT))}


