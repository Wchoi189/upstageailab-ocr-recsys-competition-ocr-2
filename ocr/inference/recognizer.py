"""Text recognition module for OCR pipeline.

This module provides the TextRecognizer class for recognizing text from
cropped and rectified text regions detected by the detection pipeline.

The module is designed for pluggable backends (TrOCR, CRNN, PaddleOCR)
with a stub implementation for initial scaffolding.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)


class RecognizerBackend(str, Enum):
    """Available recognition backends."""

    STUB = "stub"
    TROCR = "trocr"
    CRNN = "crnn"
    PADDLEOCR = "paddleocr"


@dataclass
class RecognitionInput:
    """Input to text recognizer.

    Attributes:
        crop: Rectified text crop as numpy array (H, W, C) in BGR format
        polygon: Original polygon coordinates in original image space
        detection_confidence: Confidence score from detection (0-1)
        metadata: Optional additional metadata dict
    """

    crop: np.ndarray
    polygon: np.ndarray
    detection_confidence: float
    metadata: dict | None = None

    def __post_init__(self) -> None:
        """Validate input dimensions."""
        if self.crop.ndim != 3:
            raise ValueError(f"Crop must be 3D (H, W, C), got shape {self.crop.shape}")
        if self.polygon.ndim != 2 or self.polygon.shape[1] != 2:
            raise ValueError(f"Polygon must be Nx2, got shape {self.polygon.shape}")
        if not 0.0 <= self.detection_confidence <= 1.0:
            raise ValueError(f"Detection confidence must be 0-1, got {self.detection_confidence}")


@dataclass
class RecognitionOutput:
    """Output from text recognizer.

    Attributes:
        text: Recognized text string
        confidence: Overall recognition confidence (0-1)
        char_probs: Optional per-character probabilities
        alternatives: Optional list of alternative recognition results
    """

    text: str
    confidence: float
    char_probs: list[float] | None = None
    alternatives: list[tuple[str, float]] | None = None

    def __post_init__(self) -> None:
        """Validate output values."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")


@dataclass
class RecognizerConfig:
    """Configuration for text recognizer.

    Attributes:
        backend: Recognition backend to use
        model_path: Path to recognition model (None for stub)
        max_batch_size: Maximum batch size for inference
        target_height: Target height for text crops (standard: 32)
        language: Target language code (e.g., "ko", "en")
        device: Device for inference ("cuda" or "cpu", auto-detected if None)
    """

    backend: RecognizerBackend = RecognizerBackend.STUB
    model_path: Path | str | None = None
    max_batch_size: int = 32
    target_height: int = 32
    language: str = "ko"
    device: str | None = None


class BaseRecognizer(ABC):
    """Abstract base class for text recognizers."""

    @abstractmethod
    def recognize_single(self, input_data: RecognitionInput) -> RecognitionOutput:
        """Recognize text from a single crop.

        Args:
            input_data: Recognition input with crop and metadata

        Returns:
            Recognition output with text and confidence
        """
        ...

    @abstractmethod
    def recognize_batch(self, inputs: list[RecognitionInput]) -> list[RecognitionOutput]:
        """Recognize text from multiple crops.

        Args:
            inputs: List of recognition inputs

        Returns:
            List of recognition outputs in same order as inputs
        """
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the recognizer model is loaded."""
        ...


class StubRecognizer(BaseRecognizer):
    """Stub recognizer for testing and scaffolding.

    Returns placeholder text with synthetic confidence scores.
    """

    def __init__(self, config: RecognizerConfig | None = None) -> None:
        """Initialize stub recognizer.

        Args:
            config: Optional configuration (ignored for stub)
        """
        self._config = config or RecognizerConfig()
        LOGGER.info("Initialized StubRecognizer (placeholder implementation)")

    def recognize_single(self, input_data: RecognitionInput) -> RecognitionOutput:
        """Return placeholder recognition result.

        Args:
            input_data: Recognition input (used for confidence calculation)

        Returns:
            Placeholder RecognitionOutput
        """
        # Generate placeholder text based on crop dimensions
        h, w = input_data.crop.shape[:2]
        estimated_chars = max(1, w // 20)  # Rough estimate: 20px per char

        placeholder_text = f"Text_{estimated_chars}chars"

        # Use detection confidence as proxy for recognition confidence
        confidence = input_data.detection_confidence * 0.9  # Slightly lower

        return RecognitionOutput(
            text=placeholder_text,
            confidence=confidence,
            char_probs=None,
            alternatives=None,
        )

    def recognize_batch(self, inputs: list[RecognitionInput]) -> list[RecognitionOutput]:
        """Recognize batch by calling single recognition.

        Args:
            inputs: List of recognition inputs

        Returns:
            List of recognition outputs
        """
        return [self.recognize_single(inp) for inp in inputs]

    def is_loaded(self) -> bool:
        """Stub is always 'loaded'."""
        return True


@dataclass
class TextRecognizer:
    """Text recognition engine interface.

    This class provides the main entry point for text recognition,
    delegating to backend-specific implementations.

    Example:
        >>> recognizer = TextRecognizer(config=RecognizerConfig(backend="stub"))
        >>> outputs = recognizer.recognize_batch(inputs)
    """

    config: RecognizerConfig = field(default_factory=RecognizerConfig)
    _backend: BaseRecognizer | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the backend recognizer."""
        self._backend = self._create_backend()

    def _create_backend(self) -> BaseRecognizer:
        """Create the appropriate backend recognizer.

        Returns:
            Initialized recognizer backend

        Raises:
            NotImplementedError: If backend is not yet implemented
        """
        if self.config.backend == RecognizerBackend.STUB:
            return StubRecognizer(self.config)
        elif self.config.backend == RecognizerBackend.TROCR:
            raise NotImplementedError(
                "TrOCR backend not yet implemented. "
                "Use backend='stub' for testing."
            )
        elif self.config.backend == RecognizerBackend.CRNN:
            raise NotImplementedError(
                "CRNN backend not yet implemented. "
                "Use backend='stub' for testing."
            )
        elif self.config.backend == RecognizerBackend.PADDLEOCR:
            from ocr.inference.backends.paddleocr_recognizer import PaddleOCRRecognizer

            return PaddleOCRRecognizer(self.config)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")

    def recognize_single(self, input_data: RecognitionInput) -> RecognitionOutput:
        """Recognize text from a single crop.

        Args:
            input_data: Recognition input with crop and metadata

        Returns:
            Recognition output with text and confidence
        """
        if self._backend is None:
            raise RuntimeError("Recognizer backend not initialized")
        return self._backend.recognize_single(input_data)

    def recognize_batch(self, inputs: list[RecognitionInput]) -> list[RecognitionOutput]:
        """Recognize text from multiple crops.

        Batches inputs according to max_batch_size for efficiency.

        Args:
            inputs: List of recognition inputs

        Returns:
            List of recognition outputs in same order as inputs
        """
        if self._backend is None:
            raise RuntimeError("Recognizer backend not initialized")

        if not inputs:
            return []

        # Process in batches for efficiency
        results: list[RecognitionOutput] = []
        batch_size = self.config.max_batch_size

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            batch_results = self._backend.recognize_batch(batch)
            results.extend(batch_results)

        return results

    def is_loaded(self) -> bool:
        """Check if the recognizer model is loaded."""
        return self._backend is not None and self._backend.is_loaded()

    def cleanup(self) -> None:
        """Clean up resources."""
        self._backend = None
        LOGGER.info("TextRecognizer cleanup complete")


__all__ = [
    "RecognitionInput",
    "RecognitionOutput",
    "RecognizerConfig",
    "RecognizerBackend",
    "TextRecognizer",
    "BaseRecognizer",
    "StubRecognizer",
]
