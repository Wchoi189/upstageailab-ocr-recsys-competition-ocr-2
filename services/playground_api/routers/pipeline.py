"""Pipeline preview + fallback endpoints."""

from __future__ import annotations

import base64
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...utils.paths import PROJECT_ROOT

router = APIRouter()


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

    return PipelinePreviewResponse(
        status="accepted",
        job_id=f"{request.pipeline_id}-{cache_key[-6:]}",
        routed_backend=backend,
        cache_key=cache_key,
        notes=notes,
    )


@router.post("/fallback", response_model=PipelineFallbackResponse)
def queue_fallback(request: PipelineFallbackRequest) -> PipelineFallbackResponse:
    """Placeholder backend fallback endpoint (Hooked up in Track C once worker saturation triggers)."""
    image_file = (PROJECT_ROOT / request.image_path).resolve()
    if not image_file.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_path}")

    notes = [
        "Backend execution not wired in yet.",
        "When implemented, store outputs under outputs/playground/{pipeline_id}/",
    ]
    return PipelineFallbackResponse(
        status="accepted",
        routed_backend="server-rembg",
        result_path=None,
        notes=notes,
    )


