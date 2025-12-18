"""Checkpoint discovery router for Playground Console Backend

Handles checkpoint metadata retrieval and listing.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Resolve checkpoint root relative to project root (4 levels up from routers dir)
ROUTERS_DIR = Path(__file__).parent
PROJECT_ROOT = ROUTERS_DIR.parent.parent.parent.parent
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs/experiments/train/ocr"


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


def discover_checkpoints(limit: int = 100) -> list[Checkpoint]:
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
            with open(meta_path) as f:
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
