"""Checkpoint discovery router for Playground Console Backend

Handles checkpoint metadata retrieval and listing.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

DEFAULT_CHECKPOINT_ROOT = Path(os.getenv("OCR_CHECKPOINT_PATH", "outputs/experiments/train/ocr"))


class Checkpoint(BaseModel):
    """Checkpoint metadata for UI selection."""

    checkpoint_path: str
    display_name: str
    size_mb: float
    modified_at: str


def discover_checkpoints(limit: int = 100) -> list[Checkpoint]:
    """Discover available checkpoints in outputs/experiments/train/ocr.

    Args:
        limit: Maximum number of checkpoints to return

    Returns:
        List of Checkpoint objects sorted by modification time (newest first)
    """
    if not DEFAULT_CHECKPOINT_ROOT.exists():
        logger.warning("Checkpoint root missing: %s", DEFAULT_CHECKPOINT_ROOT)
        return []

    ckpts = sorted(
        DEFAULT_CHECKPOINT_ROOT.rglob("*.ckpt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    results: list[Checkpoint] = []
    for p in ckpts[:limit]:
        stat = p.stat()
        display_name = str(p.relative_to(DEFAULT_CHECKPOINT_ROOT))
        results.append(
            Checkpoint(
                checkpoint_path=str(p),
                display_name=display_name,
                size_mb=round(stat.st_size / (1024 * 1024), 2),
                modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            )
        )
    return results


@router.get("", response_model=list[Checkpoint])
async def list_checkpoints(limit: int = 100):
    """List available OCR checkpoints.

    Args:
        limit: Maximum number of checkpoints to return (default: 100)

    Returns:
        List of checkpoint metadata sorted by modification time
    """
    checkpoints = discover_checkpoints(limit=limit)
    return checkpoints
