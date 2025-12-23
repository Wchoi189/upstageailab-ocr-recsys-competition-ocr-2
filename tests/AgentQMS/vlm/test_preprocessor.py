"""Tests for image preprocessor."""

from PIL import Image

from AgentQMS.vlm.core.preprocessor import VLMImagePreprocessor


class TestVLMImagePreprocessor:
    """Tests for VLMImagePreprocessor."""

    def test_preprocess_image(self, tmp_path):
        """Test preprocessing an image."""
        # Create a test image
        image_file = tmp_path / "test.jpg"
        img = Image.new("RGB", (3000, 2000), color="red")
        img.save(image_file, "JPEG")

        preprocessor = VLMImagePreprocessor()
        processed = preprocessor.preprocess(image_file, max_resolution=2048)

        assert processed.width <= 2048
        assert processed.height <= 2048
        assert processed.original_width == 3000
        assert processed.original_height == 2000
        assert processed.resize_ratio <= 1.0
        assert processed.base64_encoded is not None

    def test_preprocess_batch(self, tmp_path):
        """Test batch preprocessing."""
        # Create multiple test images
        image_files = []
        for i in range(3):
            image_file = tmp_path / f"test_{i}.jpg"
            img = Image.new("RGB", (1000, 1000), color="blue")
            img.save(image_file, "JPEG")
            image_files.append(image_file)

        preprocessor = VLMImagePreprocessor()
        processed_images = preprocessor.preprocess_batch(image_files, max_resolution=2048)

        assert len(processed_images) == 3
        for processed in processed_images:
            assert processed.width <= 2048
            assert processed.height <= 2048
