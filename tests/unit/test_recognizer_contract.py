"""Unit tests for text recognition module contracts.

Tests the shape and validation of RecognitionInput and RecognitionOutput
to ensure contract stability.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocr.inference.recognizer import (
    RecognitionInput,
    RecognitionOutput,
    RecognizerBackend,
    RecognizerConfig,
    StubRecognizer,
    TextRecognizer,
)


class TestRecognitionInput:
    """Tests for RecognitionInput dataclass validation."""

    def test_valid_input_creation(self):
        """Valid inputs should be accepted."""
        crop = np.zeros((32, 100, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])

        input_data = RecognitionInput(
            crop=crop,
            polygon=polygon,
            detection_confidence=0.85,
        )

        assert input_data.crop.shape == (32, 100, 3)
        assert input_data.polygon.shape == (4, 2)
        assert input_data.detection_confidence == 0.85
        assert input_data.metadata is None

    def test_valid_input_with_metadata(self):
        """Metadata field should be optional."""
        crop = np.zeros((32, 100, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])

        input_data = RecognitionInput(
            crop=crop,
            polygon=polygon,
            detection_confidence=0.9,
            metadata={"source": "test"},
        )

        assert input_data.metadata == {"source": "test"}

    def test_invalid_crop_dimensions(self):
        """2D crop should raise ValueError."""
        crop = np.zeros((32, 100), dtype=np.uint8)  # Missing channel dimension
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])

        with pytest.raises(ValueError, match="3D"):
            RecognitionInput(
                crop=crop,
                polygon=polygon,
                detection_confidence=0.5,
            )

    def test_invalid_polygon_shape(self):
        """1D polygon should raise ValueError."""
        crop = np.zeros((32, 100, 3), dtype=np.uint8)
        polygon = np.array([0, 0, 100, 0, 100, 32, 0, 32])  # Flat, not Nx2

        with pytest.raises(ValueError, match="Nx2"):
            RecognitionInput(
                crop=crop,
                polygon=polygon,
                detection_confidence=0.5,
            )

    def test_invalid_confidence_too_high(self):
        """Confidence > 1.0 should raise ValueError."""
        crop = np.zeros((32, 100, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])

        with pytest.raises(ValueError, match="0-1"):
            RecognitionInput(
                crop=crop,
                polygon=polygon,
                detection_confidence=1.5,
            )

    def test_invalid_confidence_negative(self):
        """Confidence < 0.0 should raise ValueError."""
        crop = np.zeros((32, 100, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])

        with pytest.raises(ValueError, match="0-1"):
            RecognitionInput(
                crop=crop,
                polygon=polygon,
                detection_confidence=-0.1,
            )


class TestRecognitionOutput:
    """Tests for RecognitionOutput dataclass validation."""

    def test_valid_output_creation(self):
        """Valid outputs should be accepted."""
        output = RecognitionOutput(
            text="Hello",
            confidence=0.95,
        )

        assert output.text == "Hello"
        assert output.confidence == 0.95
        assert output.char_probs is None
        assert output.alternatives is None

    def test_valid_output_with_char_probs(self):
        """char_probs field should be optional."""
        output = RecognitionOutput(
            text="Hi",
            confidence=0.9,
            char_probs=[0.95, 0.92],
        )

        assert output.char_probs == [0.95, 0.92]

    def test_valid_output_with_alternatives(self):
        """alternatives field should be optional."""
        output = RecognitionOutput(
            text="Hello",
            confidence=0.9,
            alternatives=[("Hallo", 0.7), ("Helllo", 0.5)],
        )

        assert len(output.alternatives) == 2
        assert output.alternatives[0] == ("Hallo", 0.7)

    def test_invalid_confidence_too_high(self):
        """Confidence > 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="0-1"):
            RecognitionOutput(text="Test", confidence=1.1)

    def test_invalid_confidence_negative(self):
        """Confidence < 0.0 should raise ValueError."""
        with pytest.raises(ValueError, match="0-1"):
            RecognitionOutput(text="Test", confidence=-0.1)

    def test_empty_text_allowed(self):
        """Empty string should be valid."""
        output = RecognitionOutput(text="", confidence=0.0)
        assert output.text == ""


class TestRecognizerConfig:
    """Tests for RecognizerConfig defaults."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = RecognizerConfig()

        assert config.backend == RecognizerBackend.STUB
        assert config.model_path is None
        assert config.max_batch_size == 32
        assert config.target_height == 32
        assert config.language == "ko"
        assert config.device is None

    def test_custom_config(self):
        """Custom config values should be preserved."""
        config = RecognizerConfig(
            backend=RecognizerBackend.TROCR,
            model_path="/path/to/model",
            max_batch_size=16,
            language="en",
        )

        assert config.backend == RecognizerBackend.TROCR
        assert config.model_path == "/path/to/model"
        assert config.max_batch_size == 16
        assert config.language == "en"


class TestStubRecognizer:
    """Tests for StubRecognizer placeholder implementation."""

    def test_stub_single_recognition(self):
        """Stub should return placeholder text."""
        recognizer = StubRecognizer()
        crop = np.zeros((32, 100, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])

        input_data = RecognitionInput(
            crop=crop,
            polygon=polygon,
            detection_confidence=0.9,
        )

        output = recognizer.recognize_single(input_data)

        assert isinstance(output, RecognitionOutput)
        assert isinstance(output.text, str)
        assert len(output.text) > 0
        assert 0.0 <= output.confidence <= 1.0

    def test_stub_batch_recognition(self):
        """Stub should handle batch recognition."""
        recognizer = StubRecognizer()

        inputs = []
        for _ in range(5):
            crop = np.zeros((32, 80, 3), dtype=np.uint8)
            polygon = np.array([[0, 0], [80, 0], [80, 32], [0, 32]])
            inputs.append(
                RecognitionInput(
                    crop=crop,
                    polygon=polygon,
                    detection_confidence=0.85,
                )
            )

        outputs = recognizer.recognize_batch(inputs)

        assert len(outputs) == 5
        for output in outputs:
            assert isinstance(output, RecognitionOutput)

    def test_stub_is_always_loaded(self):
        """Stub recognizer is always 'loaded'."""
        recognizer = StubRecognizer()
        assert recognizer.is_loaded() is True


class TestTextRecognizer:
    """Tests for TextRecognizer main interface."""

    def test_default_uses_stub(self):
        """Default config should use stub backend."""
        recognizer = TextRecognizer()

        assert recognizer.is_loaded()

    def test_batch_recognition_with_stub(self):
        """TextRecognizer should delegate to stub."""
        recognizer = TextRecognizer(config=RecognizerConfig(backend=RecognizerBackend.STUB))

        inputs = []
        for i in range(3):
            crop = np.zeros((32, 50 + i * 20, 3), dtype=np.uint8)
            polygon = np.array([[0, 0], [50 + i * 20, 0], [50 + i * 20, 32], [0, 32]])
            inputs.append(
                RecognitionInput(
                    crop=crop,
                    polygon=polygon,
                    detection_confidence=0.7 + i * 0.1,
                )
            )

        outputs = recognizer.recognize_batch(inputs)

        assert len(outputs) == 3
        for output in outputs:
            assert isinstance(output.text, str)
            assert 0.0 <= output.confidence <= 1.0

    def test_empty_batch(self):
        """Empty batch should return empty list."""
        recognizer = TextRecognizer()
        outputs = recognizer.recognize_batch([])
        assert outputs == []

    def test_cleanup(self):
        """Cleanup should clear backend."""
        recognizer = TextRecognizer()
        assert recognizer.is_loaded()

        recognizer.cleanup()
        assert not recognizer.is_loaded()

    def test_trocr_not_implemented(self):
        """TrOCR backend should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="TrOCR"):
            TextRecognizer(config=RecognizerConfig(backend=RecognizerBackend.TROCR))

    def test_crnn_not_implemented(self):
        """CRNN backend should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="CRNN"):
            TextRecognizer(config=RecognizerConfig(backend=RecognizerBackend.CRNN))

    def test_paddleocr_not_implemented(self):
        """PaddleOCR backend should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="PaddleOCR"):
            TextRecognizer(config=RecognizerConfig(backend=RecognizerBackend.PADDLEOCR))
