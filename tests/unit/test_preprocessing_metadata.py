"""Unit tests for preprocessing metadata utilities."""

import pytest

from ocr.inference.preprocessing_metadata import (
    calculate_padding,
    calculate_resize_dimensions,
    create_preprocessing_metadata,
    get_content_area,
)


class TestCreatePreprocessingMetadata:
    """Tests for create_preprocessing_metadata function."""

    def test_square_image_no_padding(self):
        """Test metadata for square image at target size."""
        meta = create_preprocessing_metadata((640, 640), target_size=640)

        assert meta["original_size"] == (640, 640)
        assert meta["processed_size"] == (640, 640)
        assert meta["padding"] == {"top": 0, "bottom": 0, "left": 0, "right": 0}
        assert meta["padding_position"] == "top_left"
        assert meta["content_area"] == (640, 640)
        assert meta["scale"] == 1.0
        assert meta["coordinate_system"] == "pixel"

    def test_portrait_image_width_padding(self):
        """Test metadata for portrait image (taller than wide)."""
        meta = create_preprocessing_metadata((800, 400), target_size=640)

        assert meta["original_size"] == (400, 800)
        assert meta["processed_size"] == (640, 640)
        assert meta["padding"] == {"top": 0, "bottom": 0, "left": 0, "right": 320}
        assert meta["padding_position"] == "top_left"
        assert meta["content_area"] == (320, 640)
        assert meta["scale"] == 0.8
        assert meta["coordinate_system"] == "pixel"

    def test_landscape_image_height_padding(self):
        """Test metadata for landscape image (wider than tall)."""
        meta = create_preprocessing_metadata((400, 800), target_size=640)

        assert meta["original_size"] == (800, 400)
        assert meta["processed_size"] == (640, 640)
        assert meta["padding"] == {"top": 0, "bottom": 320, "left": 0, "right": 0}
        assert meta["padding_position"] == "top_left"
        assert meta["content_area"] == (640, 320)
        assert meta["scale"] == 0.8
        assert meta["coordinate_system"] == "pixel"

    def test_large_image_downscaling(self):
        """Test metadata for large image requiring downscaling."""
        meta = create_preprocessing_metadata((1920, 1080), target_size=640)

        assert meta["original_size"] == (1080, 1920)
        assert meta["processed_size"] == (640, 640)
        # scale = 640 / 1920 = 1/3
        # resized: 360x640
        # padding: right=280
        assert meta["padding"] == {"top": 0, "bottom": 0, "left": 0, "right": 280}
        assert meta["content_area"] == (360, 640)
        assert abs(meta["scale"] - (640.0 / 1920.0)) < 1e-6
        assert meta["coordinate_system"] == "pixel"

    def test_small_image_upscaling(self):
        """Test metadata for small image requiring upscaling."""
        meta = create_preprocessing_metadata((100, 100), target_size=640)

        assert meta["original_size"] == (100, 100)
        assert meta["processed_size"] == (640, 640)
        assert meta["padding"] == {"top": 0, "bottom": 0, "left": 0, "right": 0}
        assert meta["content_area"] == (640, 640)
        assert meta["scale"] == 6.4
        assert meta["coordinate_system"] == "pixel"

    def test_shape_with_channels(self):
        """Test that function handles shape with channel dimension."""
        meta = create_preprocessing_metadata((800, 400, 3), target_size=640)

        assert meta["original_size"] == (400, 800)
        assert meta["content_area"] == (320, 640)

    def test_custom_target_size(self):
        """Test metadata with custom target size."""
        meta = create_preprocessing_metadata((1000, 800), target_size=512)

        assert meta["original_size"] == (800, 1000)
        assert meta["processed_size"] == (512, 512)
        # scale = 512 / 1000 = 0.512
        # resized: 410x512 (rounded)
        # padding: right=102
        assert meta["padding"]["right"] == 102
        assert meta["padding"]["bottom"] == 0
        assert abs(meta["scale"] - 0.512) < 1e-6

    def test_invalid_shape_raises_error(self):
        """Test that invalid shapes raise ValueError."""
        with pytest.raises(ValueError):
            create_preprocessing_metadata((0, 640), target_size=640)

        with pytest.raises(ValueError):
            create_preprocessing_metadata((640, 0), target_size=640)

        with pytest.raises(ValueError):
            create_preprocessing_metadata((640,), target_size=640)

    def test_metadata_structure_completeness(self):
        """Test that all expected metadata fields are present."""
        meta = create_preprocessing_metadata((800, 600), target_size=640)

        required_fields = [
            "original_size",
            "processed_size",
            "padding",
            "padding_position",
            "content_area",
            "scale",
            "coordinate_system",
        ]

        for field in required_fields:
            assert field in meta, f"Missing required field: {field}"

        # Check padding structure
        padding_fields = ["top", "bottom", "left", "right"]
        for field in padding_fields:
            assert field in meta["padding"], f"Missing padding field: {field}"

    def test_top_left_padding_consistency(self):
        """Test that top and left padding are always 0 (top_left position)."""
        test_shapes = [(800, 600), (600, 800), (1920, 1080), (400, 300)]

        for shape in test_shapes:
            meta = create_preprocessing_metadata(shape, target_size=640)
            assert meta["padding"]["top"] == 0, f"Top padding should be 0 for shape {shape}"
            assert meta["padding"]["left"] == 0, f"Left padding should be 0 for shape {shape}"
            assert meta["padding_position"] == "top_left"


class TestCalculateResizeDimensions:
    """Tests for calculate_resize_dimensions function."""

    def test_square_image_no_resize(self):
        """Test resize dimensions for square image at target size."""
        h, w, scale = calculate_resize_dimensions((640, 640), target_size=640)
        assert h == 640
        assert w == 640
        assert scale == 1.0

    def test_portrait_image_resize(self):
        """Test resize dimensions for portrait image."""
        h, w, scale = calculate_resize_dimensions((800, 400), target_size=640)
        assert h == 640
        assert w == 320
        assert scale == 0.8

    def test_landscape_image_resize(self):
        """Test resize dimensions for landscape image."""
        h, w, scale = calculate_resize_dimensions((400, 800), target_size=640)
        assert h == 320
        assert w == 640
        assert scale == 0.8

    def test_downscaling(self):
        """Test resize dimensions for large image."""
        h, w, scale = calculate_resize_dimensions((1920, 1080), target_size=640)
        # scale = 640 / 1920 = 1/3
        expected_scale = 640.0 / 1920.0
        assert abs(scale - expected_scale) < 1e-6
        assert h == 640
        assert w == 360  # round(1080 * scale)

    def test_upscaling(self):
        """Test resize dimensions for small image."""
        h, w, scale = calculate_resize_dimensions((100, 80), target_size=640)
        # scale = 640 / 100 = 6.4
        assert scale == 6.4
        assert h == 640
        assert w == 512  # round(80 * 6.4)

    def test_invalid_shape_raises_error(self):
        """Test that invalid shapes raise ValueError."""
        with pytest.raises(ValueError):
            calculate_resize_dimensions((0, 640), target_size=640)


class TestCalculatePadding:
    """Tests for calculate_padding function."""

    def test_square_image_no_padding(self):
        """Test padding for square image at target size."""
        pad_h, pad_w = calculate_padding((640, 640), target_size=640)
        assert pad_h == 0
        assert pad_w == 0

    def test_portrait_image_width_padding(self):
        """Test padding for portrait image."""
        pad_h, pad_w = calculate_padding((800, 400), target_size=640)
        assert pad_h == 0  # No height padding
        assert pad_w == 320  # Width padding on right

    def test_landscape_image_height_padding(self):
        """Test padding for landscape image."""
        pad_h, pad_w = calculate_padding((400, 800), target_size=640)
        assert pad_h == 320  # Height padding on bottom
        assert pad_w == 0  # No width padding

    def test_large_image_padding(self):
        """Test padding for downscaled image."""
        pad_h, pad_w = calculate_padding((1920, 1080), target_size=640)
        # scale = 640 / 1920
        # resized: 640x360
        # padding: right=280
        assert pad_h == 0
        assert pad_w == 280

    def test_small_image_no_padding(self):
        """Test padding for upscaled square image."""
        pad_h, pad_w = calculate_padding((100, 100), target_size=640)
        assert pad_h == 0
        assert pad_w == 0

    def test_invalid_shape_raises_error(self):
        """Test that invalid shapes raise ValueError."""
        with pytest.raises(ValueError):
            calculate_padding((0, 640), target_size=640)


class TestGetContentArea:
    """Tests for get_content_area function."""

    def test_square_image_full_content(self):
        """Test content area for square image at target size."""
        w, h = get_content_area((640, 640), target_size=640)
        assert w == 640
        assert h == 640

    def test_portrait_image_content(self):
        """Test content area for portrait image."""
        w, h = get_content_area((800, 400), target_size=640)
        assert w == 320
        assert h == 640

    def test_landscape_image_content(self):
        """Test content area for landscape image."""
        w, h = get_content_area((400, 800), target_size=640)
        assert w == 640
        assert h == 320

    def test_downscaled_image_content(self):
        """Test content area for large downscaled image."""
        w, h = get_content_area((1920, 1080), target_size=640)
        # scale = 640 / 1920
        # resized: 360x640
        assert w == 360
        assert h == 640

    def test_upscaled_image_content(self):
        """Test content area for small upscaled image."""
        w, h = get_content_area((100, 100), target_size=640)
        assert w == 640
        assert h == 640

    def test_invalid_shape_raises_error(self):
        """Test that invalid shapes raise ValueError."""
        with pytest.raises(ValueError):
            get_content_area((0, 640), target_size=640)


class TestIntegrationConsistency:
    """Integration tests to verify consistency across functions."""

    def test_metadata_matches_individual_calculations(self):
        """Test that metadata dict matches individual function results."""
        test_cases = [
            (640, 640),
            (800, 400),
            (400, 800),
            (1920, 1080),
            (100, 100),
        ]

        for original_h, original_w in test_cases:
            original_shape = (original_h, original_w)
            target_size = 640

            meta = create_preprocessing_metadata(original_shape, target_size)
            h, w, scale = calculate_resize_dimensions(original_shape, target_size)
            pad_h, pad_w = calculate_padding(original_shape, target_size)
            content_w, content_h = get_content_area(original_shape, target_size)

            # Check consistency
            assert meta["content_area"] == (content_w, content_h)
            assert meta["padding"]["bottom"] == pad_h
            assert meta["padding"]["right"] == pad_w
            assert meta["scale"] == scale
            assert meta["original_size"] == (original_w, original_h)

    def test_content_plus_padding_equals_target(self):
        """Test that content_area + padding == target_size."""
        test_shapes = [(800, 600), (600, 800), (1920, 1080), (400, 300), (640, 640)]

        for shape in test_shapes:
            meta = create_preprocessing_metadata(shape, target_size=640)

            content_w, content_h = meta["content_area"]
            pad_right = meta["padding"]["right"]
            pad_bottom = meta["padding"]["bottom"]

            assert content_w + pad_right == 640
            assert content_h + pad_bottom == 640
