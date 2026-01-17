"""
Core Interface Schemas (The Bridge Layer)

This module defines domain-agnostic data contracts that allow domains to communicate
without direct dependencies. All domains must use these schemas at boundaries.

Architecture Rules:
- NO domain-specific imports allowed
- Schemas must be serializable
- Must support both detection and recognition workflows
"""
from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray


@dataclass
class Box:
    """
    Generic bounding box representation.

    Supports both axis-aligned (AABB) and polygon formats.
    All coordinates are in absolute pixel space.
    """
    points: NDArray[np.float32]  # Shape: (4, 2) for quad or (2, 2) for AABB
    confidence: float = 1.0
    label: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_polygon(self) -> bool:
        """Check if box is a polygon (4 points) or AABB (2 points)."""
        return self.points.shape[0] == 4

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging/storage."""
        return {
            "points": self.points.tolist(),
            "confidence": self.confidence,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class DetectionResult:
    """
    Output from detection domain.

    Contains bounding boxes and optional confidence maps.
    """
    boxes: list[Box]
    image_shape: tuple[int, int]  # (H, W)
    confidence_map: NDArray[np.float32] | None = None
    metadata: dict[str, Any] | None = None

    def __len__(self) -> int:
        return len(self.boxes)


@dataclass
class RecognitionResult:
    """
    Output from recognition domain.

    Contains recognized text and confidence scores.
    """
    text: str
    confidence: float
    box: Box | None = None  # Associated bounding box if available
    metadata: dict[str, Any] | None = None


@dataclass
class PageResult:
    """
    Complete OCR result for a single page/image.

    Combines detection and recognition results in a unified format.
    """
    image_path: str
    image_shape: tuple[int, int]  # (H, W)
    detections: DetectionResult | None = None
    recognitions: list[RecognitionResult] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete result."""
        return {
            "image_path": self.image_path,
            "image_shape": self.image_shape,
            "detections": {
                "boxes": [b.to_dict() for b in self.detections.boxes],
                "metadata": self.detections.metadata,
            } if self.detections else None,
            "recognitions": [
                {
                    "text": r.text,
                    "confidence": r.confidence,
                    "box": r.box.to_dict() if r.box else None,
                    "metadata": r.metadata,
                }
                for r in self.recognitions
            ] if self.recognitions else None,
            "metadata": self.metadata,
        }
