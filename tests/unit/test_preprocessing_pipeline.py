"""Unit tests for preprocessing pipeline."""

import numpy as np
import pytest

try:
    import torch
    from torchvision import transforms
except ImportError:
    torch = None
    transforms = None

# Skip all tests if torch/transforms not available
pytestmark = pytest.mark.skipif(torch is None or transforms is None, reason="Torch or torchvision not available")


@pytest.fixture
def sample_transform():
    """Create a sample torchvision transform."""
    if transforms is None:
        pytest.skip("Torchvision not available")

    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@pytest.fixture
def sample_image_bgr():
    """Create a sample BGR image."""
    # Create 400x300 BGR image
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    image[:, :] = [100, 150, 200]  # BGR values
    return image


class TestPreprocessingPipelineInit:
    """Tests for PreprocessingPipeline initialization."""

    def test_init_with_defaults(self, sample_transform):
        """Test pipeline initialization with defaults."""
        from ocr.inference.preprocessing_pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(transform=sample_transform)
        assert pipeline._transform is sample_transform
        assert pipeline._target_size == 640

    def test_init_with_custom_target_size(self, sample_transform):
        """Test pipeline initialization with custom target size."""
        from ocr.inference.preprocessing_pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(transform=sample_transform, target_size=512)
        assert pipeline._target_size == 512

    def test_set_transform(self, sample_transform):
        """Test setting transform after initialization."""
        from ocr.inference.preprocessing_pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        pipeline.set_transform(sample_transform)
        assert pipeline._transform is sample_transform

    def test_set_target_size(self, sample_transform):
        """Test setting target size after initialization."""
        from ocr.inference.preprocessing_pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(transform=sample_transform)
        pipeline.set_target_size(512)
        assert pipeline._target_size == 512


class TestPreprocessingPipelineProcess:
    """Tests for PreprocessingPipeline.process method."""

    def test_process_basic(self, sample_transform, sample_image_bgr):
        """Test basic preprocessing without perspective correction."""
        from ocr.inference.preprocessing_pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(transform=sample_transform, target_size=640)
        result = pipeline.process(sample_image_bgr)

        assert result is not None
        assert result.batch is not None
        assert result.batch.shape == (1, 3, 640, 640)  # Batch, channels, H, W
        assert result.preview_image.shape == (640, 640, 3)  # H, W, channels (BGR)
        assert result.original_shape == (300, 400, 3)
        assert result.metadata is not None
        assert result.perspective_matrix is None
        assert result.original_image is None

    def test_process_with_metadata(self, sample_transform, sample_image_bgr):
        """Test that metadata is correctly generated."""
        from ocr.inference.preprocessing_pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(transform=sample_transform, target_size=640)
        result = pipeline.process(sample_image_bgr)

        assert result is not None
        assert result.metadata is not None
        assert "original_size" in result.metadata
        assert "processed_size" in result.metadata
        assert "scale" in result.metadata
        assert "padding" in result.metadata
        assert result.metadata["original_size"] == (400, 300)  # W, H
        assert result.metadata["processed_size"] == (640, 640)

    def test_process_without_transform_fails(self, sample_image_bgr):
        """Test that processing without transform fails gracefully."""
        from ocr.inference.preprocessing_pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()  # No transform
        result = pipeline.process(sample_image_bgr)

        assert result is None

    def test_process_with_custom_target_size(self, sample_transform, sample_image_bgr):
        """Test preprocessing with custom target size."""
        from ocr.inference.preprocessing_pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(transform=sample_transform, target_size=512)
        result = pipeline.process(sample_image_bgr)

        assert result is not None
        assert result.batch.shape == (1, 3, 512, 512)
        assert result.preview_image.shape == (512, 512, 3)


class TestPreprocessingPipelineOriginalDisplay:
    """Tests for process_for_original_display method."""

    def test_process_for_original_display(self, sample_transform, sample_image_bgr):
        """Test preprocessing original image for display."""
        from ocr.inference.preprocessing_pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(transform=sample_transform, target_size=640)
        result = pipeline.process_for_original_display(sample_image_bgr)

        assert result is not None
        preview_image, metadata = result
        assert preview_image.shape == (640, 640, 3)
        assert metadata is not None
        assert metadata["original_size"] == (400, 300)

    def test_process_for_original_display_without_transform(self, sample_image_bgr):
        """Test that original display fails without transform."""
        from ocr.inference.preprocessing_pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        result = pipeline.process_for_original_display(sample_image_bgr)

        assert result is None


class TestPreprocessingResult:
    """Tests for PreprocessingResult dataclass."""

    def test_preprocessing_result_attributes(self):
        """Test PreprocessingResult dataclass attributes."""
        from ocr.inference.preprocessing_pipeline import PreprocessingResult

        batch = np.zeros((1, 3, 640, 640))
        preview = np.zeros((640, 640, 3), dtype=np.uint8)
        original_shape = (300, 400, 3)
        metadata = {"test": "data"}

        result = PreprocessingResult(
            batch=batch,
            preview_image=preview,
            original_shape=original_shape,
            metadata=metadata,
        )

        assert result.batch.shape == (1, 3, 640, 640)
        assert result.preview_image.shape == (640, 640, 3)
        assert result.original_shape == (300, 400, 3)
        assert result.metadata == {"test": "data"}
        assert result.perspective_matrix is None
        assert result.original_image is None

    def test_preprocessing_result_with_perspective(self):
        """Test PreprocessingResult with perspective correction data."""
        from ocr.inference.preprocessing_pipeline import PreprocessingResult

        batch = np.zeros((1, 3, 640, 640))
        preview = np.zeros((640, 640, 3), dtype=np.uint8)
        original_shape = (300, 400, 3)
        metadata = {}
        perspective_matrix = np.eye(3, dtype=np.float32)
        original_image = np.zeros((300, 400, 3), dtype=np.uint8)

        result = PreprocessingResult(
            batch=batch,
            preview_image=preview,
            original_shape=original_shape,
            metadata=metadata,
            perspective_matrix=perspective_matrix,
            original_image=original_image,
        )

        assert result.perspective_matrix is not None
        assert result.perspective_matrix.shape == (3, 3)
        assert result.original_image is not None
        assert result.original_image.shape == (300, 400, 3)
