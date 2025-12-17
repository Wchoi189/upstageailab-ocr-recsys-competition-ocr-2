"""Unit tests for preview generation utilities."""

import base64

import cv2
import numpy as np
import pytest

from ocr.inference.preview_generator import PreviewGenerator, create_preview_with_metadata


class TestPreviewGeneratorInit:
    """Tests for PreviewGenerator initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        generator = PreviewGenerator()
        assert generator.jpeg_quality == 85
        assert generator._encode_params == [cv2.IMWRITE_JPEG_QUALITY, 85]

    def test_custom_jpeg_quality(self):
        """Test initialization with custom JPEG quality."""
        generator = PreviewGenerator(jpeg_quality=95)
        assert generator.jpeg_quality == 95
        assert generator._encode_params == [cv2.IMWRITE_JPEG_QUALITY, 95]

    def test_minimum_jpeg_quality(self):
        """Test initialization with minimum JPEG quality."""
        generator = PreviewGenerator(jpeg_quality=0)
        assert generator.jpeg_quality == 0

    def test_maximum_jpeg_quality(self):
        """Test initialization with maximum JPEG quality."""
        generator = PreviewGenerator(jpeg_quality=100)
        assert generator.jpeg_quality == 100

    def test_invalid_jpeg_quality_too_low(self):
        """Test that invalid JPEG quality raises error."""
        with pytest.raises(ValueError, match="JPEG quality must be between 0 and 100"):
            PreviewGenerator(jpeg_quality=-1)

    def test_invalid_jpeg_quality_too_high(self):
        """Test that invalid JPEG quality raises error."""
        with pytest.raises(ValueError, match="JPEG quality must be between 0 and 100"):
            PreviewGenerator(jpeg_quality=101)


class TestEncodePreviewImage:
    """Tests for encode_preview_image method."""

    def test_encode_jpg_default(self):
        """Test encoding image as JPEG."""
        generator = PreviewGenerator(jpeg_quality=85)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = generator.encode_preview_image(image, format="jpg")

        assert result is not None
        assert isinstance(result, str)
        # Verify it's valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_encode_jpeg_format(self):
        """Test encoding with 'jpeg' format string."""
        generator = PreviewGenerator(jpeg_quality=85)
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = generator.encode_preview_image(image, format="jpeg")

        assert result is not None
        assert isinstance(result, str)

    def test_encode_png_format(self):
        """Test encoding image as PNG."""
        generator = PreviewGenerator()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        result = generator.encode_preview_image(image, format="png")

        assert result is not None
        assert isinstance(result, str)
        # PNG should generally be larger than JPEG for same image
        decoded = base64.b64decode(result)
        assert len(decoded) > 0

    def test_encode_colored_image(self):
        """Test encoding a colored image."""
        generator = PreviewGenerator()
        # Create a simple colored image
        image = np.full((100, 100, 3), fill_value=[255, 128, 64], dtype=np.uint8)

        result = generator.encode_preview_image(image)

        assert result is not None
        assert isinstance(result, str)

    def test_encode_none_image(self):
        """Test that encoding None image returns None."""
        generator = PreviewGenerator()

        result = generator.encode_preview_image(None, format="jpg")

        assert result is None

    def test_encode_empty_image(self):
        """Test that encoding empty image returns None."""
        generator = PreviewGenerator()
        image = np.array([])

        result = generator.encode_preview_image(image, format="jpg")

        assert result is None

    def test_encode_unsupported_format(self):
        """Test that unsupported format raises error."""
        generator = PreviewGenerator()
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported format"):
            generator.encode_preview_image(image, format="bmp")

    def test_jpeg_quality_affects_size(self):
        """Test that JPEG quality affects encoded size."""
        image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

        # High quality should produce larger file
        gen_high = PreviewGenerator(jpeg_quality=95)
        result_high = gen_high.encode_preview_image(image)
        size_high = len(base64.b64decode(result_high))

        # Low quality should produce smaller file
        gen_low = PreviewGenerator(jpeg_quality=50)
        result_low = gen_low.encode_preview_image(image)
        size_low = len(base64.b64decode(result_low))

        assert size_high > size_low


class TestAttachPreviewToPayload:
    """Tests for attach_preview_to_payload method."""

    def test_attach_preview_simple(self):
        """Test attaching preview to simple payload."""
        generator = PreviewGenerator()
        payload = {"polygons": "0 0 100 0 100 100 0 100", "texts": ["Text1"]}
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        result = generator.attach_preview_to_payload(payload, image, transform_polygons=False)

        assert "preview_image_base64" in result
        assert isinstance(result["preview_image_base64"], str)
        assert result["polygons"] == payload["polygons"]  # No transformation
        assert result["texts"] == payload["texts"]

    def test_attach_preview_with_metadata(self):
        """Test attaching preview with metadata."""
        generator = PreviewGenerator()
        payload = {"polygons": "0 0 100 0 100 100 0 100"}
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        metadata = {
            "original_size": (640, 640),
            "processed_size": (640, 640),
            "coordinate_system": "pixel",
        }

        result = generator.attach_preview_to_payload(payload, image, metadata=metadata, transform_polygons=False)

        assert "preview_image_base64" in result
        assert "meta" in result
        assert result["meta"] == metadata

    def test_attach_preview_without_metadata(self):
        """Test attaching preview without metadata logs warning."""
        generator = PreviewGenerator()
        payload = {"polygons": "0 0 100 0 100 100 0 100"}
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        result = generator.attach_preview_to_payload(payload, image, metadata=None, transform_polygons=False)

        assert "preview_image_base64" in result
        assert "meta" not in result

    def test_attach_preview_transforms_polygons(self):
        """Test that polygon transformation is applied when requested."""
        generator = PreviewGenerator()
        payload = {"polygons": "0 0 100 0 100 100 0 100"}
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        original_shape = (800, 400)  # Portrait image

        result = generator.attach_preview_to_payload(
            payload,
            image,
            transform_polygons=True,
            original_shape=original_shape,
            target_size=640,
        )

        assert "preview_image_base64" in result
        # Polygons should be transformed (scaled by 0.8 for 800->640)
        assert result["polygons"] != payload["polygons"]
        assert result["polygons"] == "0 0 80 0 80 80 0 80"

    def test_attach_preview_skips_polygon_transform_without_shape(self):
        """Test that polygon transformation is skipped if original_shape not provided."""
        generator = PreviewGenerator()
        payload = {"polygons": "0 0 100 0 100 100 0 100"}
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        result = generator.attach_preview_to_payload(
            payload,
            image,
            transform_polygons=True,
            original_shape=None,  # No shape provided
        )

        assert "preview_image_base64" in result
        # Polygons should not be transformed
        assert result["polygons"] == payload["polygons"]

    def test_attach_preview_non_dict_payload(self):
        """Test that non-dict payload is returned unchanged."""
        generator = PreviewGenerator()
        payload = "not a dict"
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        result = generator.attach_preview_to_payload(payload, image)

        assert result == payload

    def test_attach_preview_empty_polygons(self):
        """Test attaching preview with empty polygons."""
        generator = PreviewGenerator()
        payload = {"polygons": "", "texts": []}
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        result = generator.attach_preview_to_payload(
            payload,
            image,
            transform_polygons=True,
            original_shape=(640, 640),
        )

        assert "preview_image_base64" in result
        assert result["polygons"] == ""

    def test_attach_preview_preserves_original_payload(self):
        """Test that original payload is not modified."""
        generator = PreviewGenerator()
        payload = {"polygons": "0 0 100 0 100 100 0 100", "texts": ["Text1"]}
        original_payload = payload.copy()
        image = np.zeros((640, 640, 3), dtype=np.uint8)

        result = generator.attach_preview_to_payload(payload, image, transform_polygons=False)

        assert payload == original_payload  # Original not modified
        assert result != payload  # Result is different object


class TestCreatePreviewWithMetadata:
    """Tests for create_preview_with_metadata convenience function."""

    def test_create_preview_basic(self):
        """Test basic preview creation."""
        payload = {"polygons": "0 0 100 0 100 100 0 100"}
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        metadata = {"original_size": (640, 640), "processed_size": (640, 640)}

        result = create_preview_with_metadata(
            payload,
            image,
            metadata=metadata,
            original_shape=(640, 640),
        )

        assert "preview_image_base64" in result
        assert "meta" in result
        assert result["meta"] == metadata

    def test_create_preview_with_custom_quality(self):
        """Test preview creation with custom JPEG quality."""
        payload = {"polygons": "0 0 100 0 100 100 0 100"}
        image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)

        result_high = create_preview_with_metadata(
            payload,
            image,
            original_shape=(640, 640),
            jpeg_quality=95,
        )

        result_low = create_preview_with_metadata(
            payload,
            image,
            original_shape=(640, 640),
            jpeg_quality=50,
        )

        # High quality should produce larger base64 string
        assert len(result_high["preview_image_base64"]) > len(result_low["preview_image_base64"])

    def test_create_preview_transforms_polygons(self):
        """Test that polygons are transformed by default."""
        payload = {"polygons": "0 0 100 0 100 100 0 100"}
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        original_shape = (800, 400)  # Portrait image, scale = 0.8

        result = create_preview_with_metadata(
            payload,
            image,
            original_shape=original_shape,
        )

        assert "preview_image_base64" in result
        # Polygons should be scaled by 0.8
        assert result["polygons"] == "0 0 80 0 80 80 0 80"

    def test_create_preview_with_all_parameters(self):
        """Test preview creation with all parameters."""
        payload = {"polygons": "0 0 200 0 200 200 0 200", "texts": ["Text1"]}
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        metadata = {
            "original_size": (1024, 768),
            "processed_size": (512, 512),
            "coordinate_system": "pixel",
        }

        result = create_preview_with_metadata(
            payload=payload,
            preview_image=image,
            metadata=metadata,
            original_shape=(1024, 768),
            target_size=512,
            jpeg_quality=90,
        )

        assert "preview_image_base64" in result
        assert "meta" in result
        assert result["meta"] == metadata
        assert result["texts"] == ["Text1"]


class TestTransformPolygonsToPreviewSpace:
    """Tests for _transform_polygons_to_preview_space method."""

    def test_transform_polygons_portrait_image(self):
        """Test polygon transformation for portrait image."""
        generator = PreviewGenerator()
        payload = {"polygons": "0 0 100 0 100 100 0 100"}

        result = generator._transform_polygons_to_preview_space(
            payload,
            original_shape=(800, 400),  # height, width
            target_size=640,
        )

        # Scale should be 640/800 = 0.8
        assert result["polygons"] == "0 0 80 0 80 80 0 80"

    def test_transform_polygons_landscape_image(self):
        """Test polygon transformation for landscape image."""
        generator = PreviewGenerator()
        payload = {"polygons": "0 0 100 0 100 100 0 100"}

        result = generator._transform_polygons_to_preview_space(
            payload,
            original_shape=(400, 800),  # height, width
            target_size=640,
        )

        # Scale should be 640/800 = 0.8
        assert result["polygons"] == "0 0 80 0 80 80 0 80"

    def test_transform_polygons_empty_payload(self):
        """Test transformation with empty polygon string."""
        generator = PreviewGenerator()
        payload = {"polygons": ""}

        result = generator._transform_polygons_to_preview_space(
            payload,
            original_shape=(640, 640),
        )

        assert result["polygons"] == ""

    def test_transform_polygons_no_polygons_key(self):
        """Test transformation with payload missing polygons key."""
        generator = PreviewGenerator()
        payload = {"texts": ["Text1"]}

        result = generator._transform_polygons_to_preview_space(
            payload,
            original_shape=(640, 640),
        )

        assert result == payload  # Unchanged

    def test_transform_polygons_non_dict_payload(self):
        """Test transformation with non-dict payload."""
        generator = PreviewGenerator()
        payload = "not a dict"

        result = generator._transform_polygons_to_preview_space(
            payload,
            original_shape=(640, 640),
        )

        assert result == payload  # Unchanged
