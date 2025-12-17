"""Unit tests for coordinate transformation utilities."""

import numpy as np
import pytest

from ocr.inference.coordinate_manager import (
    CoordinateTransformationManager,
    TransformMetadata,
    calculate_transform_metadata,
    compute_forward_scales,
    compute_inverse_matrix,
    transform_polygon_to_processed_space,
    transform_polygons_string_to_processed_space,
)


class TestCalculateTransformMetadata:
    """Tests for calculate_transform_metadata function."""

    def test_square_image_no_padding(self):
        """Test square image that exactly fits target size."""
        metadata = calculate_transform_metadata((640, 640), target_size=640)
        assert metadata.original_h == 640
        assert metadata.original_w == 640
        assert metadata.resized_h == 640
        assert metadata.resized_w == 640
        assert metadata.target_size == 640
        assert metadata.scale == 1.0
        assert metadata.pad_h == 0
        assert metadata.pad_w == 0

    def test_portrait_image_pad_width(self):
        """Test portrait image (taller than wide) - needs width padding."""
        metadata = calculate_transform_metadata((800, 400), target_size=640)
        assert metadata.original_h == 800
        assert metadata.original_w == 400
        # scale = 640 / 800 = 0.8
        # resized_h = round(800 * 0.8) = 640
        # resized_w = round(400 * 0.8) = 320
        assert metadata.resized_h == 640
        assert metadata.resized_w == 320
        assert metadata.scale == 0.8
        assert metadata.pad_h == 0  # No height padding
        assert metadata.pad_w == 320  # Width padding = 640 - 320

    def test_landscape_image_pad_height(self):
        """Test landscape image (wider than tall) - needs height padding."""
        metadata = calculate_transform_metadata((400, 800), target_size=640)
        assert metadata.original_h == 400
        assert metadata.original_w == 800
        # scale = 640 / 800 = 0.8
        # resized_h = round(400 * 0.8) = 320
        # resized_w = round(800 * 0.8) = 640
        assert metadata.resized_h == 320
        assert metadata.resized_w == 640
        assert metadata.scale == 0.8
        assert metadata.pad_h == 320  # Height padding = 640 - 320
        assert metadata.pad_w == 0  # No width padding

    def test_small_image_upscaling(self):
        """Test small image that needs upscaling."""
        metadata = calculate_transform_metadata((100, 100), target_size=640)
        assert metadata.original_h == 100
        assert metadata.original_w == 100
        assert metadata.resized_h == 640
        assert metadata.resized_w == 640
        assert metadata.scale == 6.4
        assert metadata.pad_h == 0
        assert metadata.pad_w == 0

    def test_large_image_downscaling(self):
        """Test large image that needs downscaling."""
        metadata = calculate_transform_metadata((1920, 1080), target_size=640)
        assert metadata.original_h == 1920
        assert metadata.original_w == 1080
        # scale = 640 / 1920 = 0.333...
        scale = 640.0 / 1920.0
        assert abs(metadata.scale - scale) < 1e-6
        assert metadata.resized_h == 640
        assert metadata.resized_w == 360  # round(1080 * 0.333...) = 360
        assert metadata.pad_h == 0
        assert metadata.pad_w == 280  # 640 - 360

    def test_shape_with_channels(self):
        """Test that function handles shape with channel dimension."""
        metadata = calculate_transform_metadata((800, 400, 3), target_size=640)
        assert metadata.original_h == 800
        assert metadata.original_w == 400
        assert metadata.resized_h == 640
        assert metadata.resized_w == 320

    def test_invalid_shape_too_short(self):
        """Test invalid shape with insufficient dimensions."""
        with pytest.raises(ValueError, match="must have at least 2 dimensions"):
            calculate_transform_metadata((640,), target_size=640)

    def test_invalid_dimensions_zero(self):
        """Test invalid dimensions with zero values."""
        with pytest.raises(ValueError, match="Invalid original dimensions"):
            calculate_transform_metadata((0, 640), target_size=640)

        with pytest.raises(ValueError, match="Invalid original dimensions"):
            calculate_transform_metadata((640, 0), target_size=640)

    def test_invalid_dimensions_negative(self):
        """Test invalid dimensions with negative values."""
        with pytest.raises(ValueError, match="Invalid original dimensions"):
            calculate_transform_metadata((-640, 480), target_size=640)


class TestComputeInverseMatrix:
    """Tests for compute_inverse_matrix function."""

    def test_square_image_identity_scale(self):
        """Test inverse matrix for square image at target size."""
        matrix = compute_inverse_matrix((640, 640), target_size=640)
        expected = np.eye(3, dtype=np.float32)
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_downscaled_image_inverse(self):
        """Test inverse matrix for downscaled image."""
        # Original: 1920x1080, Target: 640
        # scale = 640 / 1920 = 1/3
        # inv_scale = 3.0
        matrix = compute_inverse_matrix((1920, 1080), target_size=640)
        scale = 640.0 / 1920.0
        inv_scale = 1.0 / scale
        expected = np.array(
            [[inv_scale, 0.0, 0.0], [0.0, inv_scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_upscaled_image_inverse(self):
        """Test inverse matrix for upscaled image."""
        # Original: 100x100, Target: 640
        # scale = 640 / 100 = 6.4
        # inv_scale = 1 / 6.4
        matrix = compute_inverse_matrix((100, 100), target_size=640)
        scale = 640.0 / 100.0
        inv_scale = 1.0 / scale
        expected = np.array(
            [[inv_scale, 0.0, 0.0], [0.0, inv_scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_no_translation_top_left_padding(self):
        """Test that inverse matrix has zero translation (top_left padding)."""
        matrix = compute_inverse_matrix((800, 400), target_size=640)
        # Translation components should be zero
        assert matrix[0, 2] == 0.0
        assert matrix[1, 2] == 0.0

    def test_invalid_shape_returns_identity(self):
        """Test that invalid shapes return identity matrix."""
        matrix = compute_inverse_matrix((0, 640), target_size=640)
        np.testing.assert_array_equal(matrix, np.eye(3, dtype=np.float32))

        matrix = compute_inverse_matrix((640, 0), target_size=640)
        np.testing.assert_array_equal(matrix, np.eye(3, dtype=np.float32))


class TestComputeForwardScales:
    """Tests for compute_forward_scales function."""

    def test_square_image_unit_scales(self):
        """Test forward scales for square image at target size."""
        scale_x, scale_y = compute_forward_scales((640, 640), target_size=640)
        assert abs(scale_x - 1.0) < 1e-6
        assert abs(scale_y - 1.0) < 1e-6

    def test_portrait_image_different_scales(self):
        """Test forward scales for portrait image."""
        # Original: 800x400, Target: 640
        # scale = 640 / 800 = 0.8
        # resized: 640x320
        # scale_x = 320 / 400 = 0.8
        # scale_y = 640 / 800 = 0.8
        scale_x, scale_y = compute_forward_scales((800, 400), target_size=640)
        assert abs(scale_x - 0.8) < 1e-6
        assert abs(scale_y - 0.8) < 1e-6

    def test_landscape_image_different_scales(self):
        """Test forward scales for landscape image."""
        # Original: 400x800, Target: 640
        # scale = 640 / 800 = 0.8
        # resized: 320x640
        # scale_x = 640 / 800 = 0.8
        # scale_y = 320 / 400 = 0.8
        scale_x, scale_y = compute_forward_scales((400, 800), target_size=640)
        assert abs(scale_x - 0.8) < 1e-6
        assert abs(scale_y - 0.8) < 1e-6

    def test_invalid_shape_returns_identity(self):
        """Test that invalid shapes return identity scales."""
        scale_x, scale_y = compute_forward_scales((0, 640), target_size=640)
        assert scale_x == 1.0
        assert scale_y == 1.0


class TestTransformPolygonToProcessedSpace:
    """Tests for transform_polygon_to_processed_space function."""

    def test_transform_square_polygon_no_scaling(self):
        """Test transforming polygon with no scaling needed."""
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        transformed = transform_polygon_to_processed_space(polygon, (640, 640), target_size=640)

        expected = polygon  # No scaling for 640x640 image
        np.testing.assert_array_almost_equal(transformed, expected)

    def test_transform_polygon_with_downscaling(self):
        """Test transforming polygon with downscaling."""
        # Original: 1920x1080, Target: 640
        # scale = 640 / 1920 = 1/3
        # resized: 640x360
        # scale_x = 360 / 1080 = 1/3
        # scale_y = 640 / 1920 = 1/3
        polygon = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float32)
        transformed = transform_polygon_to_processed_space(polygon, (1920, 1080), target_size=640)

        scale_x = 360.0 / 1080.0
        scale_y = 640.0 / 1920.0
        expected = polygon * np.array([scale_x, scale_y], dtype=np.float32)
        np.testing.assert_array_almost_equal(transformed, expected, decimal=4)

    def test_transform_polygon_with_upscaling(self):
        """Test transforming polygon with upscaling."""
        # Original: 100x100, Target: 640
        # scale = 640 / 100 = 6.4
        # scale_x = scale_y = 6.4
        polygon = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], dtype=np.float32)
        transformed = transform_polygon_to_processed_space(polygon, (100, 100), target_size=640)

        expected = polygon * 6.4
        np.testing.assert_array_almost_equal(transformed, expected, decimal=4)


class TestTransformPolygonsStringToProcessedSpace:
    """Tests for transform_polygons_string_to_processed_space function."""

    def test_empty_string(self):
        """Test transforming empty polygon string."""
        result = transform_polygons_string_to_processed_space("", (640, 640), target_size=640)
        assert result == ""

    def test_single_polygon_no_scaling(self):
        """Test transforming single polygon with no scaling."""
        polygons_str = "0 0 100 0 100 100 0 100"
        result = transform_polygons_string_to_processed_space(polygons_str, (640, 640), target_size=640)

        # With scale=1.0, coordinates should be unchanged (rounded to int)
        assert result == "0 0 100 0 100 100 0 100"

    def test_single_polygon_with_scaling(self):
        """Test transforming single polygon with scaling."""
        # Original: 800x400, Target: 640
        # scale = 640 / 800 = 0.8
        # resized: 640x320
        # scale_x = scale_y = 0.8
        polygons_str = "0 0 100 0 100 100 0 100"
        result = transform_polygons_string_to_processed_space(polygons_str, (800, 400), target_size=640)

        # Expected: all coordinates * 0.8 = 80
        assert result == "0 0 80 0 80 80 0 80"

    def test_multiple_polygons(self):
        """Test transforming multiple polygons separated by pipe."""
        polygons_str = "0 0 100 0 100 100 0 100|200 200 300 200 300 300 200 300"
        result = transform_polygons_string_to_processed_space(polygons_str, (640, 640), target_size=640)

        # No scaling, should be unchanged
        assert result == "0 0 100 0 100 100 0 100|200 200 300 200 300 300 200 300"

    def test_polygon_with_floating_point_coords(self):
        """Test transforming polygon with floating point coordinates."""
        polygons_str = "0.5 0.5 100.5 0.5 100.5 100.5 0.5 100.5"
        result = transform_polygons_string_to_processed_space(polygons_str, (640, 640), target_size=640)

        # Should round to nearest integer
        assert result == "0 0 100 0 100 100 0 100"

    def test_invalid_polygon_too_few_coords(self):
        """Test that polygons with fewer than 6 coordinates are skipped."""
        polygons_str = "0 0 100 0"  # Only 2 points (4 coords), need at least 3 points
        result = transform_polygons_string_to_processed_space(polygons_str, (640, 640), target_size=640)

        # Should return empty string
        assert result == ""

    def test_mixed_valid_invalid_polygons(self):
        """Test mix of valid and invalid polygons."""
        polygons_str = "0 0 100 0|0 0 100 0 100 100 0 100|200 200"
        result = transform_polygons_string_to_processed_space(polygons_str, (640, 640), target_size=640)

        # Only the second polygon is valid (6+ coords)
        assert result == "0 0 100 0 100 100 0 100"

    def test_invalid_original_shape(self):
        """Test with invalid original shape."""
        polygons_str = "0 0 100 0 100 100 0 100"
        result = transform_polygons_string_to_processed_space(polygons_str, (0, 640), target_size=640)

        # Should return empty string for invalid shape
        assert result == ""


class TestCoordinateTransformationManager:
    """Tests for CoordinateTransformationManager class."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = CoordinateTransformationManager(target_size=640)
        assert manager.target_size == 640
        assert manager.metadata is None

    def test_set_original_shape(self):
        """Test setting original shape and caching metadata."""
        manager = CoordinateTransformationManager(target_size=640)
        manager.set_original_shape((800, 400))

        assert manager.metadata is not None
        assert manager.metadata.original_h == 800
        assert manager.metadata.original_w == 400
        assert manager.metadata.resized_h == 640
        assert manager.metadata.resized_w == 320

    def test_metadata_caching(self):
        """Test that metadata is cached for same shape."""
        manager = CoordinateTransformationManager(target_size=640)
        manager.set_original_shape((800, 400))
        metadata1 = manager.metadata

        # Set same shape again
        manager.set_original_shape((800, 400))
        metadata2 = manager.metadata

        # Should be same object (cached)
        assert metadata1 is metadata2

    def test_metadata_cache_invalidation(self):
        """Test that metadata cache is invalidated for different shape."""
        manager = CoordinateTransformationManager(target_size=640)
        manager.set_original_shape((800, 400))
        metadata1 = manager.metadata

        # Set different shape
        manager.set_original_shape((640, 640))
        metadata2 = manager.metadata

        # Should be different objects
        assert metadata1 is not metadata2
        assert metadata2.original_h == 640
        assert metadata2.original_w == 640

    def test_get_inverse_matrix_with_shape(self):
        """Test getting inverse matrix with explicit shape."""
        manager = CoordinateTransformationManager(target_size=640)
        matrix = manager.get_inverse_matrix((800, 400))

        scale = 640.0 / 800.0
        inv_scale = 1.0 / scale
        expected = np.array(
            [[inv_scale, 0.0, 0.0], [0.0, inv_scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_get_inverse_matrix_cached(self):
        """Test getting inverse matrix from cached metadata."""
        manager = CoordinateTransformationManager(target_size=640)
        manager.set_original_shape((800, 400))
        matrix = manager.get_inverse_matrix()

        scale = 640.0 / 800.0
        inv_scale = 1.0 / scale
        expected = np.array(
            [[inv_scale, 0.0, 0.0], [0.0, inv_scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_almost_equal(matrix, expected)

    def test_get_inverse_matrix_no_cache_raises(self):
        """Test that getting inverse matrix without cache raises error."""
        manager = CoordinateTransformationManager(target_size=640)
        with pytest.raises(ValueError, match="No original shape set"):
            manager.get_inverse_matrix()

    def test_get_forward_scales_with_shape(self):
        """Test getting forward scales with explicit shape."""
        manager = CoordinateTransformationManager(target_size=640)
        scale_x, scale_y = manager.get_forward_scales((800, 400))

        assert abs(scale_x - 0.8) < 1e-6
        assert abs(scale_y - 0.8) < 1e-6

    def test_get_forward_scales_cached(self):
        """Test getting forward scales from cached metadata."""
        manager = CoordinateTransformationManager(target_size=640)
        manager.set_original_shape((800, 400))
        scale_x, scale_y = manager.get_forward_scales()

        assert abs(scale_x - 0.8) < 1e-6
        assert abs(scale_y - 0.8) < 1e-6

    def test_get_forward_scales_no_cache_raises(self):
        """Test that getting forward scales without cache raises error."""
        manager = CoordinateTransformationManager(target_size=640)
        with pytest.raises(ValueError, match="No original shape set"):
            manager.get_forward_scales()

    def test_transform_polygon_forward_with_shape(self):
        """Test transforming polygon with explicit shape."""
        manager = CoordinateTransformationManager(target_size=640)
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        transformed = manager.transform_polygon_forward(polygon, (800, 400))

        expected = polygon * 0.8
        np.testing.assert_array_almost_equal(transformed, expected)

    def test_transform_polygon_forward_cached(self):
        """Test transforming polygon using cached metadata."""
        manager = CoordinateTransformationManager(target_size=640)
        manager.set_original_shape((800, 400))
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        transformed = manager.transform_polygon_forward(polygon)

        expected = polygon * 0.8
        np.testing.assert_array_almost_equal(transformed, expected)

    def test_transform_polygon_forward_no_cache_raises(self):
        """Test that transforming polygon without cache raises error."""
        manager = CoordinateTransformationManager(target_size=640)
        polygon = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        with pytest.raises(ValueError, match="No original shape set"):
            manager.transform_polygon_forward(polygon)

    def test_transform_polygons_string_forward_with_shape(self):
        """Test transforming polygon string with explicit shape."""
        manager = CoordinateTransformationManager(target_size=640)
        polygons_str = "0 0 100 0 100 100 0 100"
        result = manager.transform_polygons_string_forward(polygons_str, (800, 400))

        assert result == "0 0 80 0 80 80 0 80"

    def test_transform_polygons_string_forward_cached(self):
        """Test transforming polygon string using cached metadata."""
        manager = CoordinateTransformationManager(target_size=640)
        manager.set_original_shape((800, 400))
        polygons_str = "0 0 100 0 100 100 0 100"
        result = manager.transform_polygons_string_forward(polygons_str)

        assert result == "0 0 80 0 80 80 0 80"

    def test_transform_polygons_string_forward_no_cache_raises(self):
        """Test that transforming polygon string without cache raises error."""
        manager = CoordinateTransformationManager(target_size=640)
        polygons_str = "0 0 100 0 100 100 0 100"
        with pytest.raises(ValueError, match="No original shape set"):
            manager.transform_polygons_string_forward(polygons_str)
