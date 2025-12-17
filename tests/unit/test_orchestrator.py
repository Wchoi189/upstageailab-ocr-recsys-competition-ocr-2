"""Unit tests for inference orchestrator."""

import numpy as np
import pytest


class TestInferenceOrchestratorInit:
    """Tests for InferenceOrchestrator initialization."""

    def test_init_with_default_device(self):
        """Test orchestrator initialization with auto-detected device."""
        from ocr.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        assert orchestrator.model_manager is not None
        assert orchestrator.model_manager.device in ("cuda", "cpu")
        assert orchestrator.preprocessing_pipeline is None  # Not initialized until model loaded
        assert orchestrator.postprocessing_pipeline is None
        assert orchestrator.preview_generator is not None

    def test_init_with_explicit_device(self):
        """Test orchestrator initialization with explicit device."""
        from ocr.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator(device="cpu")
        assert orchestrator.model_manager.device == "cpu"

    def test_context_manager(self):
        """Test orchestrator as context manager."""
        from ocr.inference.orchestrator import InferenceOrchestrator

        with InferenceOrchestrator() as orchestrator:
            assert orchestrator is not None

        # After exit, cleanup should have been called
        assert orchestrator.model_manager.model is None


class TestInferenceOrchestratorLoadModel:
    """Tests for InferenceOrchestrator.load_model method."""

    def test_load_model_fails_without_ocr_modules(self, monkeypatch):
        """Test that load_model fails gracefully without OCR modules."""
        from ocr.inference.orchestrator import InferenceOrchestrator

        # Mock OCR_MODULES_AVAILABLE
        import ocr.inference.model_manager as mm_module
        monkeypatch.setattr(mm_module, "OCR_MODULES_AVAILABLE", False)

        orchestrator = InferenceOrchestrator()
        result = orchestrator.load_model("fake_checkpoint.pth")

        assert result is False
        assert orchestrator.preprocessing_pipeline is None
        assert orchestrator.postprocessing_pipeline is None

    def test_load_model_with_nonexistent_checkpoint(self):
        """Test loading from nonexistent checkpoint."""
        from ocr.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        result = orchestrator.load_model("/nonexistent/checkpoint.pth")

        assert result is False


class TestInferenceOrchestratorPredict:
    """Tests for InferenceOrchestrator.predict method."""

    def test_predict_without_loaded_model(self):
        """Test that predict fails without loaded model."""
        from ocr.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = orchestrator.predict(image)

        assert result is None

    def test_predict_without_pipelines(self, monkeypatch):
        """Test that predict fails without initialized pipelines."""
        from ocr.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        # Mock model as loaded but pipelines not initialized
        orchestrator.model_manager.model = object()  # Fake model

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = orchestrator.predict(image)

        assert result is None


class TestInferenceOrchestratorUpdateParams:
    """Tests for InferenceOrchestrator.update_postprocessor_params method."""

    def test_update_params_without_pipeline(self):
        """Test updating parameters without initialized pipeline."""
        from ocr.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        # Should not raise, just log warning
        orchestrator.update_postprocessor_params(binarization_thresh=0.5)

    def test_update_params_without_settings(self):
        """Test updating parameters when pipeline has no settings."""
        from ocr.inference.orchestrator import InferenceOrchestrator
        from ocr.inference.postprocessing_pipeline import PostprocessingPipeline

        orchestrator = InferenceOrchestrator()
        orchestrator.postprocessing_pipeline = PostprocessingPipeline()  # No settings

        # Should not raise, just log warning
        orchestrator.update_postprocessor_params(binarization_thresh=0.5)


class TestInferenceOrchestratorCleanup:
    """Tests for InferenceOrchestrator.cleanup method."""

    def test_cleanup(self):
        """Test cleanup method."""
        from ocr.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        orchestrator.cleanup()

        # Model manager should be cleaned
        assert orchestrator.model_manager.model is None
        assert orchestrator.model_manager.config is None
