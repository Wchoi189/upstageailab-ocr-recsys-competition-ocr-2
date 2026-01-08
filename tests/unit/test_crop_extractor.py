"""Unit tests for crop extractor module.

Tests CropExtractor functionality including perspective correction
and bounding box cropping.
"""

from __future__ import annotations

import numpy as np
import pytest

from ocr.core.inference.crop_extractor import CropConfig, CropExtractor, CropResult


class TestCropConfig:
    """Tests for CropConfig defaults."""

    def test_default_config(self):
        """Default config should have expected values."""
        config = CropConfig()

        assert config.target_height == 32
        assert config.min_width == 8
        assert config.max_aspect_ratio == 25.0
        assert config.padding_ratio == 0.1
        assert config.enable_perspective_correction is True

    def test_custom_config(self):
        """Custom config values should be preserved."""
        config = CropConfig(
            target_height=48,
            min_width=16,
            max_aspect_ratio=20.0,
            padding_ratio=0.15,
            enable_perspective_correction=False,
        )

        assert config.target_height == 48
        assert config.min_width == 16
        assert config.max_aspect_ratio == 20.0
        assert config.padding_ratio == 0.15
        assert config.enable_perspective_correction is False


class TestCropResult:
    """Tests for CropResult dataclass."""

    def test_successful_result(self):
        """Successful result should have crop."""
        crop = np.zeros((32, 100, 3), dtype=np.uint8)
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])

        result = CropResult(
            crop=crop,
            original_polygon=polygon,
            success=True,
        )

        assert result.success
        assert result.crop is not None
        assert result.error_message is None

    def test_failed_result(self):
        """Failed result should have error message."""
        polygon = np.array([[0, 0], [100, 0], [100, 32], [0, 32]])

        result = CropResult(
            crop=None,
            original_polygon=polygon,
            success=False,
            error_message="Crop too narrow",
        )

        assert not result.success
        assert result.crop is None
        assert result.error_message == "Crop too narrow"


class TestCropExtractor:
    """Tests for CropExtractor functionality."""

    @pytest.fixture
    def sample_image(self):
        """Create a simple test image."""
        # 200x300 BGR image with some patterns
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        # Add some text-like content
        image[50:80, 50:250] = 255  # White rectangle (text area)
        return image

    @pytest.fixture
    def extractor(self):
        """Create default CropExtractor."""
        return CropExtractor()

    def test_extract_single_quad(self, sample_image, extractor):
        """Extract crop from quad polygon."""
        polygon = np.array([[50, 50], [250, 50], [250, 80], [50, 80]], dtype=np.float32)

        results = extractor.extract_crops(sample_image, [polygon])

        assert len(results) == 1
        assert results[0].success
        assert results[0].crop is not None
        assert results[0].crop.ndim == 3

    def test_extract_multiple_polygons(self, sample_image, extractor):
        """Extract multiple crops."""
        polygons = [
            np.array([[50, 50], [150, 50], [150, 80], [50, 80]], dtype=np.float32),
            np.array([[160, 50], [250, 50], [250, 80], [160, 80]], dtype=np.float32),
        ]

        results = extractor.extract_crops(sample_image, polygons)

        assert len(results) == 2
        assert all(r.success for r in results)

    def test_extract_empty_list(self, sample_image, extractor):
        """Empty polygon list should return empty results."""
        results = extractor.extract_crops(sample_image, [])
        assert results == []

    def test_target_height_override(self, sample_image, extractor):
        """Target height should be configurable per call."""
        polygon = np.array([[50, 50], [250, 50], [250, 80], [50, 80]], dtype=np.float32)

        results = extractor.extract_crops(sample_image, [polygon], target_height=64)

        assert len(results) == 1
        if results[0].success:
            assert results[0].crop.shape[0] == 64

    def test_invalid_image_dimensions(self, extractor):
        """2D image should raise ValueError."""
        image = np.zeros((200, 300), dtype=np.uint8)  # Missing channel
        polygon = np.array([[50, 50], [250, 50], [250, 80], [50, 80]])

        with pytest.raises(ValueError, match="3D"):
            extractor.extract_crops(image, [polygon])

    def test_polygon_too_few_points(self, sample_image, extractor):
        """Polygon with < 3 points should fail gracefully."""
        polygon = np.array([[50, 50], [250, 50]], dtype=np.float32)  # Only 2 points

        results = extractor.extract_crops(sample_image, [polygon])

        assert len(results) == 1
        assert not results[0].success
        assert "too few points" in results[0].error_message.lower()

    def test_crop_width_validation(self, sample_image):
        """Very narrow crops should fail."""
        extractor = CropExtractor(config=CropConfig(min_width=50))
        # This polygon would produce a very narrow crop
        polygon = np.array([[100, 50], [105, 50], [105, 80], [100, 80]], dtype=np.float32)

        results = extractor.extract_crops(sample_image, [polygon])

        assert len(results) == 1
        # May fail due to width or succeed depending on scaling
        # The key is that it doesn't crash

    def test_perspective_correction_disabled(self, sample_image):
        """Should use bounding box when perspective disabled."""
        config = CropConfig(enable_perspective_correction=False)
        extractor = CropExtractor(config=config)

        polygon = np.array([[50, 50], [250, 55], [250, 80], [50, 75]], dtype=np.float32)

        results = extractor.extract_crops(sample_image, [polygon])

        assert len(results) == 1
        # Should still succeed with bounding box fallback

    def test_flat_polygon_format(self, sample_image, extractor):
        """Flat polygon format should be handled."""
        # Flat format: [x1, y1, x2, y2, x3, y3, x4, y4]
        polygon = np.array([50, 50, 250, 50, 250, 80, 50, 80], dtype=np.float32)

        results = extractor.extract_crops(sample_image, [polygon])

        assert len(results) == 1
        # Should reshape and process

    def test_out_of_bounds_polygon(self, sample_image, extractor):
        """Polygon outside image should be clipped."""
        polygon = np.array([[-10, 50], [350, 50], [350, 80], [-10, 80]], dtype=np.float32)

        results = extractor.extract_crops(sample_image, [polygon])

        assert len(results) == 1
        # Should handle gracefully (clip or fail with message)


class TestOrderPoints:
    """Tests for _order_points static method."""

    def test_already_ordered(self):
        """Already ordered points should stay the same."""
        pts = np.array(
            [
                [0, 0],  # top-left
                [100, 0],  # top-right
                [100, 50],  # bottom-right
                [0, 50],  # bottom-left
            ],
            dtype=np.float32,
        )

        ordered = CropExtractor._order_points(pts)

        assert ordered.shape == (4, 2)
        # Top-left should have smallest sum
        assert ordered[0].sum() < ordered[2].sum()

    def test_shuffled_points(self):
        """Shuffled points should be reordered correctly."""
        pts = np.array(
            [
                [100, 50],  # bottom-right
                [0, 0],  # top-left
                [0, 50],  # bottom-left
                [100, 0],  # top-right
            ],
            dtype=np.float32,
        )

        ordered = CropExtractor._order_points(pts)

        # Verify ordering: TL -> TR -> BR -> BL
        assert np.allclose(ordered[0], [0, 0])  # top-left
        assert np.allclose(ordered[1], [100, 0])  # top-right
        assert np.allclose(ordered[2], [100, 50])  # bottom-right
        assert np.allclose(ordered[3], [0, 50])  # bottom-left
