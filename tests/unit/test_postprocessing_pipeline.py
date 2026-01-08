"""Unit tests for postprocessing pipeline."""

import numpy as np
import pytest

from ocr.core.inference.config_loader import PostprocessSettings


@pytest.fixture
def sample_settings():
    """Create sample postprocessing settings."""
    return PostprocessSettings(
        binarization_thresh=0.3,
        box_thresh=0.7,
        max_candidates=1000,
        min_detection_size=3,
    )


@pytest.fixture
def sample_predictions():
    """Create sample model predictions."""
    # Create a simple probability map
    prob_map = np.zeros((640, 640), dtype=np.float32)
    # Add a detection region
    prob_map[100:200, 100:200] = 0.8

    return {
        "prob_maps": prob_map[np.newaxis, :, :],  # Add batch dimension
    }


@pytest.fixture
def mock_model_with_head():
    """Create mock model with working head."""

    class MockHead:
        def get_polygons_from_maps(self, batch, predictions):
            # Return mock polygons
            boxes = [[[10, 10], [100, 10], [100, 100], [10, 100]]]
            scores = [[0.95]]
            return boxes, scores

    class MockModel:
        def __init__(self):
            self.head = MockHead()

    return MockModel()


@pytest.fixture
def mock_model_without_head():
    """Create mock model without head."""

    class MockModel:
        pass

    return MockModel()


class TestPostprocessingPipelineInit:
    """Tests for PostprocessingPipeline initialization."""

    def test_init_with_settings(self, sample_settings):
        """Test pipeline initialization with settings."""
        from ocr.core.inference.postprocessing_pipeline import PostprocessingPipeline

        pipeline = PostprocessingPipeline(settings=sample_settings)
        assert pipeline._settings is sample_settings

    def test_init_without_settings(self):
        """Test pipeline initialization without settings."""
        from ocr.core.inference.postprocessing_pipeline import PostprocessingPipeline

        pipeline = PostprocessingPipeline()
        assert pipeline._settings is None

    def test_set_settings(self, sample_settings):
        """Test setting settings after initialization."""
        from ocr.core.inference.postprocessing_pipeline import PostprocessingPipeline

        pipeline = PostprocessingPipeline()
        pipeline.set_settings(sample_settings)
        assert pipeline._settings is sample_settings


class TestPostprocessingPipelineProcess:
    """Tests for PostprocessingPipeline.process method."""

    def test_process_with_head_success(self, sample_settings, sample_predictions, mock_model_with_head):
        """Test processing with successful head-based decoding."""
        from ocr.core.inference.postprocessing_pipeline import PostprocessingPipeline

        pipeline = PostprocessingPipeline(settings=sample_settings)
        processed_tensor = np.zeros((1, 3, 640, 640))
        original_shape = (480, 640, 3)

        result = pipeline.process(
            mock_model_with_head,
            processed_tensor,
            sample_predictions,
            original_shape,
        )

        assert result is not None
        assert result.method == "head"
        assert isinstance(result.polygons, str)
        assert isinstance(result.texts, list)
        assert isinstance(result.confidences, list)

    def test_process_fallback_when_no_head(self, sample_settings, sample_predictions, mock_model_without_head):
        """Test fallback to contour-based processing when no head."""
        from ocr.core.inference.postprocessing_pipeline import PostprocessingPipeline

        pipeline = PostprocessingPipeline(settings=sample_settings)
        processed_tensor = np.zeros((1, 3, 640, 640))
        original_shape = (480, 640, 3)

        result = pipeline.process(
            mock_model_without_head,
            processed_tensor,
            sample_predictions,
            original_shape,
        )

        assert result is not None
        assert result.method == "fallback"
        assert isinstance(result.polygons, str)
        assert isinstance(result.texts, list)
        assert isinstance(result.confidences, list)

    def test_process_fails_without_settings_for_fallback(self, sample_predictions, mock_model_without_head):
        """Test that fallback fails gracefully without settings."""
        from ocr.core.inference.postprocessing_pipeline import PostprocessingPipeline

        pipeline = PostprocessingPipeline()  # No settings
        processed_tensor = np.zeros((1, 3, 640, 640))
        original_shape = (480, 640, 3)

        result = pipeline.process(
            mock_model_without_head,
            processed_tensor,
            sample_predictions,
            original_shape,
        )

        # Should fail because fallback needs settings
        assert result is None


class TestPostprocessingResult:
    """Tests for PostprocessingResult dataclass."""

    def test_postprocessing_result_attributes(self):
        """Test PostprocessingResult dataclass attributes."""
        from ocr.core.inference.postprocessing_pipeline import PostprocessingResult

        result = PostprocessingResult(
            polygons="10 20 30 20 30 40 10 40",
            texts=["Text_1"],
            confidences=[0.95],
            method="head",
        )

        assert result.polygons == "10 20 30 20 30 40 10 40"
        assert result.texts == ["Text_1"]
        assert result.confidences == [0.95]
        assert result.method == "head"

    def test_postprocessing_result_with_multiple_detections(self):
        """Test PostprocessingResult with multiple detections."""
        from ocr.core.inference.postprocessing_pipeline import PostprocessingResult

        result = PostprocessingResult(
            polygons="10 20 30 20 30 40 10 40|50 60 70 60 70 80 50 80",
            texts=["Text_1", "Text_2"],
            confidences=[0.95, 0.87],
            method="fallback",
        )

        assert "|" in result.polygons
        assert len(result.texts) == 2
        assert len(result.confidences) == 2
        assert result.method == "fallback"

    def test_postprocessing_result_empty_detections(self):
        """Test PostprocessingResult with no detections."""
        from ocr.core.inference.postprocessing_pipeline import PostprocessingResult

        result = PostprocessingResult(
            polygons="",
            texts=[],
            confidences=[],
            method="head",
        )

        assert result.polygons == ""
        assert len(result.texts) == 0
        assert len(result.confidences) == 0
