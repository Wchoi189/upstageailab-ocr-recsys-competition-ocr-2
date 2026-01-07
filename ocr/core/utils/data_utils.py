"""Shared data utilities for OCR processing."""

from typing import Any

from ocr.core.utils.config_utils import is_config


def extract_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata from a sample dictionary.

    Args:
        sample: Dictionary containing sample data and metadata.

    Returns:
        Dictionary containing extracted metadata.
    """
    metadata = sample.get("metadata")
    if metadata is None:
        return {}
    if hasattr(metadata, "model_dump"):
        try:
            return metadata.model_dump()
        except Exception:
            return dict(metadata)
    if is_config(metadata):
        return metadata
    return {}
