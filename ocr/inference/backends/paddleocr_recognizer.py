"""PaddleOCR PP-OCRv5 recognition backend implementation.

This module provides a PaddleOCR-based text recognizer that integrates
PP-OCRv5 server model for Korean text recognition.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ocr.inference.recognizer import RecognitionInput, RecognitionOutput, RecognizerConfig

from ocr.inference.recognizer import BaseRecognizer, RecognitionOutput

LOGGER = logging.getLogger(__name__)


class PaddleOCRRecognizer(BaseRecognizer):
    """PaddleOCR PP-OCRv5 recognition backend.

    This recognizer uses PaddleOCR's PP-OCRv5 server model for Korean text
    recognition. It supports GPU acceleration and batch processing.

    Attributes:
        _config: Recognizer configuration
        _ocr: PaddleOCR instance for text recognition
        _loaded: Flag indicating if model is loaded
    """

    def __init__(self, config: RecognizerConfig) -> None:
        """Initialize PaddleOCR recognizer.

        Args:
            config: Recognizer configuration with backend, language, etc.

        Raises:
            ImportError: If paddleocr is not installed
            RuntimeError: If GPU is requested but not available
        """
        self._config = config
        self._ocr = None
        self._loaded = False

        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise ImportError(
                "PaddleOCR is not installed. "
                "Install with: uv sync --extra recognition"
            ) from e

        # Configure PaddleOCR
        use_gpu = config.device == "cuda" if config.device else True

        try:
            self._ocr = PaddleOCR(
                use_gpu=use_gpu,
                lang=config.language,
                rec_model_dir=str(config.model_path) if config.model_path else None,
                use_angle_cls=False,  # Already handled by CropExtractor
                det=False,  # Recognition only
                use_space_char=True,
                show_log=False,
            )
            self._loaded = True
            LOGGER.info(
                f"Initialized PaddleOCRRecognizer "
                f"(lang={config.language}, gpu={use_gpu})"
            )
        except Exception as e:
            LOGGER.error(f"Failed to initialize PaddleOCR: {e}")
            raise RuntimeError(f"PaddleOCR initialization failed: {e}") from e

    def recognize_single(self, input_data: RecognitionInput) -> RecognitionOutput:
        """Recognize text from a single crop.

        Args:
            input_data: Recognition input with crop and metadata

        Returns:
            Recognition output with text and confidence

        Raises:
            RuntimeError: If recognizer is not loaded
        """
        if not self._loaded or self._ocr is None:
            raise RuntimeError("PaddleOCR recognizer not loaded")

        try:
            # PaddleOCR expects RGB format, but our crops are in BGR
            # Convert BGR to RGB for PaddleOCR
            crop_rgb = input_data.crop[:, :, ::-1].copy()

            # Run OCR (det=False for recognition only)
            result = self._ocr.ocr(crop_rgb, det=False, cls=False)

            # Parse result
            # PaddleOCR returns [[[text, confidence]]] for single image
            if result and result[0] and len(result[0]) > 0:
                text, confidence = result[0][0]
                return RecognitionOutput(
                    text=text if text else "",
                    confidence=float(confidence),
                )
            else:
                # Empty or failed recognition
                return RecognitionOutput(
                    text="",
                    confidence=0.0,
                )

        except Exception as e:
            LOGGER.warning(f"Recognition failed for crop: {e}")
            return RecognitionOutput(
                text="",
                confidence=0.0,
            )

    def recognize_batch(self, inputs: list[RecognitionInput]) -> list[RecognitionOutput]:
        """Recognize text from multiple crops.

        Args:
            inputs: List of recognition inputs

        Returns:
            List of recognition outputs in same order as inputs

        Raises:
            RuntimeError: If recognizer is not loaded
        """
        if not self._loaded or self._ocr is None:
            raise RuntimeError("PaddleOCR recognizer not loaded")

        if not inputs:
            return []

        try:
            # Convert all crops from BGR to RGB
            crops_rgb = [inp.crop[:, :, ::-1].copy() for inp in inputs]

            # Run batch OCR
            results = self._ocr.ocr(crops_rgb, det=False, cls=False)

            # Parse results
            outputs = []
            for result in results:
                if result and len(result) > 0:
                    text, confidence = result[0]
                    outputs.append(
                        RecognitionOutput(
                            text=text if text else "",
                            confidence=float(confidence),
                        )
                    )
                else:
                    outputs.append(
                        RecognitionOutput(
                            text="",
                            confidence=0.0,
                        )
                    )

            return outputs

        except Exception as e:
            LOGGER.error(f"Batch recognition failed: {e}")
            # Return empty results for all inputs on failure
            return [
                RecognitionOutput(text="", confidence=0.0)
                for _ in inputs
            ]

    def is_loaded(self) -> bool:
        """Check if the recognizer model is loaded.

        Returns:
            True if model is loaded and ready for inference
        """
        return self._loaded and self._ocr is not None


__all__ = ["PaddleOCRRecognizer"]
