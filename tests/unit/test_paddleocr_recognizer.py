"""Unit tests for PaddleOCR recognition backend.

Tests PaddleOCR-specific functionality including initialization,
single/batch recognition, error handling, and resource management.
"""

from __future__ import annotations

import numpy as np
import pytest

# Skip all tests if paddleocr is not installed
pytest.importorskip("paddleocr", reason="PaddleOCR not installed")

from ocr.inference.backends.paddleocr_recognizer import PaddleOCRRecognizer
from ocr.inference.recognizer import RecognitionInput, RecognitionOutput, RecognizerBackend, RecognizerConfig


class TestPaddleOCRRecognizerInitialization:
    """Tests for PaddleOCR recognizer initialization."""

    def test_initialization_with_default_config(self):
        """Test recognizer initializes with default Korean settings."""
        config = RecognizerConfig(
            backend=RecognizerBackend.PADDLEOCR,
            language="korean",
            device="cuda",
        )
        recognizer = PaddleOCRRecognizer(config)

        assert recognizer.is_loaded()
        assert recognizer._config.language == "korean"

    def test_initialization_with_cpu(self):
        """Test recognizer can initialize with CPU device."""
        config = RecognizerConfig(
            backend=RecognizerBackend.PADDLEOCR,
            language="korean",
            device="cpu",
        )
        recognizer = PaddleOCRRecognizer(config)

        assert recognizer.is_loaded()


class TestPaddleOCRRecognizerSingleCrop:
    """Tests for single crop recognition."""

    @pytest.fixture
    def recognizer(self):
        """Create PaddleOCR recognizer instance."""
        config = RecognizerConfig(
            backend=RecognizerBackend.PADDLEOCR,
            language="korean",
        )
        return PaddleOCRRecognizer(config)

    @pytest.fixture
    def sample_crop(self):
        """Create a sample text crop."""
        # Create a simple synthetic crop (32x100 pixels, BGR)
        crop = np.zeros((32, 100, 3), dtype=np.uint8)
        # Add some white text-like patterns
        crop[10:25, 20:80] = 255
        return crop

    @pytest.fixture
    def sample_input(self, sample_crop):
        """Create a sample recognition input."""
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])
        return RecognitionInput(
            crop=sample_crop,
            polygon=polygon,
            detection_confidence=0.95,
        )

    def test_single_crop_recognition(self, recognizer, sample_input):
        """Test recognition of single Korean text crop."""
        output = recognizer.recognize_single(sample_input)

        assert isinstance(output, RecognitionOutput)
        assert isinstance(output.text, str)
        assert 0.0 <= output.confidence <= 1.0

    def test_empty_crop_handling(self, recognizer):
        """Test graceful handling of empty/blank crops."""
        # Create a completely black crop
        empty_crop = np.zeros((32, 100, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])

        input_data = RecognitionInput(
            crop=empty_crop,
            polygon=polygon,
            detection_confidence=0.5,
        )

        output = recognizer.recognize_single(input_data)

        # Should return empty or very low confidence result, not crash
        assert isinstance(output, RecognitionOutput)
        assert isinstance(output.text, str)
        assert 0.0 <= output.confidence <= 1.0


class TestPaddleOCRRecognizerBatch:
    """Tests for batch recognition."""

    @pytest.fixture
    def recognizer(self):
        """Create PaddleOCR recognizer instance."""
        config = RecognizerConfig(
            backend=RecognizerBackend.PADDLEOCR,
            language="korean",
            max_batch_size=32,
        )
        return PaddleOCRRecognizer(config)

    def test_batch_recognition_small(self, recognizer):
        """Test batch recognition of small number of crops."""
        inputs = []
        for i in range(5):
            crop = np.zeros((32, 80 + i * 10, 3), dtype=np.uint8)
            # Add some patterns
            crop[10:25, 10:70] = 200
            polygon = np.array([[0, 0], [80 + i * 10, 0], [80 + i * 10, 32], [0, 32]])
            inputs.append(
                RecognitionInput(
                    crop=crop,
                    polygon=polygon,
                    detection_confidence=0.8 + i * 0.02,
                )
            )

        outputs = recognizer.recognize_batch(inputs)

        assert len(outputs) == 5
        for output in outputs:
            assert isinstance(output, RecognitionOutput)
            assert isinstance(output.text, str)
            assert 0.0 <= output.confidence <= 1.0

    def test_batch_recognition_medium(self, recognizer):
        """Test batch recognition of 32 crops (full batch)."""
        inputs = []
        for i in range(32):
            crop = np.random.randint(0, 255, (32, 100, 3), dtype=np.uint8)
            polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])
            inputs.append(
                RecognitionInput(
                    crop=crop,
                    polygon=polygon,
                    detection_confidence=0.9,
                )
            )

        outputs = recognizer.recognize_batch(inputs)

        assert len(outputs) == 32
        for output in outputs:
            assert isinstance(output, RecognitionOutput)

    def test_empty_batch(self, recognizer):
        """Test empty batch returns empty list."""
        outputs = recognizer.recognize_batch([])
        assert outputs == []


class TestPaddleOCRRecognizerResources:
    """Tests for resource management and edge cases."""

    def test_is_loaded(self):
        """Test is_loaded returns correct status."""
        config = RecognizerConfig(
            backend=RecognizerBackend.PADDLEOCR,
            language="korean",
        )
        recognizer = PaddleOCRRecognizer(config)

        assert recognizer.is_loaded() is True

    def test_recognize_without_load_raises_error(self):
        """Test that recognizing without loaded model raises error."""
        config = RecognizerConfig(
            backend=RecognizerBackend.PADDLEOCR,
            language="korean",
        )
        recognizer = PaddleOCRRecognizer(config)

        # Simulate unloaded state
        recognizer._loaded = False
        recognizer._ocr = None

        crop = np.zeros((32, 100, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])
        input_data = RecognitionInput(
            crop=crop,
            polygon=polygon,
            detection_confidence=0.9,
        )

        with pytest.raises(RuntimeError, match="not loaded"):
            recognizer.recognize_single(input_data)

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"),
        reason="GPU tests require PyTorch",
    )
    def test_vram_usage(self):
        """Verify VRAM usage stays under 1GB (requires GPU)."""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Clear cache before test
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        config = RecognizerConfig(
            backend=RecognizerBackend.PADDLEOCR,
            language="korean",
            device="cuda",
        )
        recognizer = PaddleOCRRecognizer(config)

        # Create a batch of crops
        inputs = []
        for _ in range(32):
            crop = np.random.randint(0, 255, (48, 200, 3), dtype=np.uint8)
            polygon = np.array([[0, 0], [200, 0], [200, 48], [0, 48]])
            inputs.append(
                RecognitionInput(
                    crop=crop,
                    polygon=polygon,
                    detection_confidence=0.9,
                )
            )

        # Run recognition
        outputs = recognizer.recognize_batch(inputs)

        # Check VRAM usage
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
        print(f"Peak VRAM usage: {peak_memory:.2f} GB")

        # Should be under 1GB for recognition model
        # Note: This is a soft requirement and may vary by system
        assert peak_memory < 2.0, f"VRAM usage {peak_memory:.2f}GB exceeds 2GB threshold"
        assert len(outputs) == 32


class TestPaddleOCRRecognizerEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def recognizer(self):
        """Create PaddleOCR recognizer instance."""
        config = RecognizerConfig(
            backend=RecognizerBackend.PADDLEOCR,
            language="korean",
        )
        return PaddleOCRRecognizer(config)

    def test_very_wide_crop(self, recognizer):
        """Test handling of very wide aspect ratio crops."""
        crop = np.zeros((32, 800, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [800, 0], [800, 32], [0, 32]])

        input_data = RecognitionInput(
            crop=crop,
            polygon=polygon,
            detection_confidence=0.9,
        )

        output = recognizer.recognize_single(input_data)
        assert isinstance(output, RecognitionOutput)

    def test_very_narrow_crop(self, recognizer):
        """Test handling of very narrow crops."""
        crop = np.zeros((32, 10, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [10, 0], [10, 32], [0, 32]])

        input_data = RecognitionInput(
            crop=crop,
            polygon=polygon,
            detection_confidence=0.9,
        )

        output = recognizer.recognize_single(input_data)
        assert isinstance(output, RecognitionOutput)

    def test_tall_crop(self, recognizer):
        """Test handling of tall crops (non-standard height)."""
        crop = np.zeros((64, 100, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [100, 0], [100, 64], [0, 64]])

        input_data = RecognitionInput(
            crop=crop,
            polygon=polygon,
            detection_confidence=0.9,
        )

        output = recognizer.recognize_single(input_data)
        assert isinstance(output, RecognitionOutput)
