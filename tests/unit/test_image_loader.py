"""Unit tests for image loader utilities."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ocr.core.inference.image_loader import ImageLoader, LoadedImage


class TestImageLoaderFromPath:
    """Tests for ImageLoader.load_from_path method."""

    def test_load_rgb_image(self, tmp_path: Path):
        """Test loading a simple RGB image."""
        # Create test image
        img = Image.new("RGB", (100, 200), color=(255, 0, 0))
        img_path = tmp_path / "test.png"
        img.save(img_path)

        loader = ImageLoader()
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.image.shape == (200, 100, 3)  # H, W, C
        assert loaded.orientation == 1
        assert loaded.raw_width == 100
        assert loaded.raw_height == 200
        assert loaded.canonical_width == 100
        assert loaded.canonical_height == 200

        # Verify BGR format (red in RGB should be (0, 0, 255) in BGR)
        assert np.allclose(loaded.image[0, 0], [0, 0, 255], atol=5)

    def test_load_rgba_image(self, tmp_path: Path):
        """Test loading an RGBA image (with alpha channel)."""
        # Create RGBA image
        img = Image.new("RGBA", (50, 75), color=(0, 255, 0, 255))
        img_path = tmp_path / "test_rgba.png"
        img.save(img_path)

        loader = ImageLoader()
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.image.shape == (75, 50, 3)  # Alpha channel removed
        assert loaded.orientation == 1

        # Verify BGR format (green in RGBA should be (0, 255, 0) in BGR)
        assert np.allclose(loaded.image[0, 0], [0, 255, 0], atol=5)

    def test_load_grayscale_image(self, tmp_path: Path):
        """Test loading a grayscale image."""
        # Create grayscale image
        img = Image.new("L", (60, 80), color=128)
        img_path = tmp_path / "test_gray.png"
        img.save(img_path)

        loader = ImageLoader()
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.image.shape == (80, 60, 3)  # Converted to 3-channel
        assert loaded.orientation == 1

        # All channels should be similar (grayscale converted to BGR)
        pixel = loaded.image[0, 0]
        assert np.allclose(pixel, [128, 128, 128], atol=5)

    def test_load_jpeg_image(self, tmp_path: Path):
        """Test loading a JPEG image."""
        # Create JPEG image
        img = Image.new("RGB", (120, 80), color=(100, 150, 200))
        img_path = tmp_path / "test.jpg"
        img.save(img_path, "JPEG")

        loader = ImageLoader()
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.image.shape == (80, 120, 3)
        assert loaded.orientation == 1

    def test_file_not_found(self, tmp_path: Path):
        """Test handling of non-existent file."""
        loader = ImageLoader()
        loaded = loader.load_from_path(tmp_path / "nonexistent.png")

        assert loaded is None

    def test_turbojpeg_disabled(self, tmp_path: Path):
        """Test loading with TurboJPEG disabled."""
        img = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img_path = tmp_path / "test.jpg"
        img.save(img_path, "JPEG")

        loader = ImageLoader(use_turbojpeg=False)
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.image.shape == (100, 100, 3)


class TestImageLoaderExifOrientation:
    """Tests for EXIF orientation handling."""

    def test_orientation_1_no_rotation(self, tmp_path: Path):
        """Test image with orientation 1 (normal, no rotation)."""
        # Create image with EXIF orientation 1
        img = Image.new("RGB", (100, 200), color=(255, 0, 0))
        img_path = tmp_path / "test_orient1.jpg"

        # Save with EXIF orientation 1
        exif = img.getexif()
        exif[0x0112] = 1  # Orientation tag
        img.save(img_path, "JPEG", exif=exif)

        loader = ImageLoader()
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.orientation == 1
        assert loaded.raw_width == 100
        assert loaded.raw_height == 200
        assert loaded.canonical_width == 100
        assert loaded.canonical_height == 200

    def test_orientation_6_rotate_90_cw(self, tmp_path: Path):
        """Test image with orientation 6 (rotate 90° CW)."""
        # Create image with EXIF orientation 6
        img = Image.new("RGB", (100, 200), color=(0, 255, 0))
        img_path = tmp_path / "test_orient6.jpg"

        # Save with EXIF orientation 6
        exif = img.getexif()
        exif[0x0112] = 6  # Rotate 90° CW
        img.save(img_path, "JPEG", exif=exif)

        loader = ImageLoader()
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.orientation == 6
        assert loaded.raw_width == 100
        assert loaded.raw_height == 200
        # After rotation, width and height should be swapped
        assert loaded.canonical_width == 200
        assert loaded.canonical_height == 100

    def test_orientation_3_rotate_180(self, tmp_path: Path):
        """Test image with orientation 3 (rotate 180°)."""
        img = Image.new("RGB", (150, 100), color=(0, 0, 255))
        img_path = tmp_path / "test_orient3.jpg"

        exif = img.getexif()
        exif[0x0112] = 3  # Rotate 180°
        img.save(img_path, "JPEG", exif=exif)

        loader = ImageLoader()
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.orientation == 3
        assert loaded.raw_width == 150
        assert loaded.raw_height == 100
        # 180° rotation doesn't swap dimensions
        assert loaded.canonical_width == 150
        assert loaded.canonical_height == 100

    def test_orientation_8_rotate_90_ccw(self, tmp_path: Path):
        """Test image with orientation 8 (rotate 90° CCW)."""
        img = Image.new("RGB", (120, 180), color=(255, 255, 0))
        img_path = tmp_path / "test_orient8.jpg"

        exif = img.getexif()
        exif[0x0112] = 8  # Rotate 90° CCW
        img.save(img_path, "JPEG", exif=exif)

        loader = ImageLoader()
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.orientation == 8
        assert loaded.raw_width == 120
        assert loaded.raw_height == 180
        # After rotation, dimensions should be swapped
        assert loaded.canonical_width == 180
        assert loaded.canonical_height == 120


class TestImageLoaderFromPil:
    """Tests for ImageLoader.load_from_pil method."""

    def test_load_rgb_pil_image(self):
        """Test loading from RGB PIL image."""
        pil_img = Image.new("RGB", (100, 150), color=(255, 128, 64))

        loader = ImageLoader()
        loaded = loader.load_from_pil(pil_img)

        assert loaded.image.shape == (150, 100, 3)
        assert loaded.orientation == 1
        assert loaded.raw_width == 100
        assert loaded.raw_height == 150
        assert loaded.canonical_width == 100
        assert loaded.canonical_height == 150

        # Verify BGR conversion
        assert np.allclose(loaded.image[0, 0], [64, 128, 255], atol=5)

    def test_load_rgba_pil_image(self):
        """Test loading from RGBA PIL image."""
        pil_img = Image.new("RGBA", (80, 120), color=(200, 100, 50, 255))

        loader = ImageLoader()
        loaded = loader.load_from_pil(pil_img)

        assert loaded.image.shape == (120, 80, 3)
        assert loaded.orientation == 1

    def test_load_grayscale_pil_image(self):
        """Test loading from grayscale PIL image."""
        pil_img = Image.new("L", (90, 110), color=200)

        loader = ImageLoader()
        loaded = loader.load_from_pil(pil_img)

        assert loaded.image.shape == (110, 90, 3)
        pixel = loaded.image[0, 0]
        assert np.allclose(pixel, [200, 200, 200], atol=5)

    def test_pil_with_exif_orientation(self, tmp_path: Path):
        """Test loading PIL image with EXIF orientation."""
        img = Image.new("RGB", (100, 200), color=(255, 0, 0))
        img_path = tmp_path / "test_exif.jpg"

        exif = img.getexif()
        exif[0x0112] = 6  # Rotate 90° CW
        img.save(img_path, "JPEG", exif=exif)

        # Load and pass to loader
        with Image.open(img_path) as pil_img:
            loader = ImageLoader()
            loaded = loader.load_from_pil(pil_img)

            assert loaded.orientation == 6
            assert loaded.raw_width == 100
            assert loaded.raw_height == 200
            assert loaded.canonical_width == 200
            assert loaded.canonical_height == 100


class TestImageLoaderFromArray:
    """Tests for ImageLoader.load_from_array method."""

    def test_load_bgr_array(self):
        """Test loading BGR numpy array."""
        # Create BGR array (blue image)
        bgr_array = np.zeros((100, 150, 3), dtype=np.uint8)
        bgr_array[:, :, 0] = 255  # Blue channel

        loader = ImageLoader()
        loaded = loader.load_from_array(bgr_array, color_space="BGR")

        assert loaded.image.shape == (100, 150, 3)
        assert loaded.orientation == 1
        assert loaded.raw_width == 150
        assert loaded.raw_height == 100
        assert loaded.canonical_width == 150
        assert loaded.canonical_height == 100

        # Should be unchanged (already BGR)
        assert np.array_equal(loaded.image, bgr_array)

    def test_load_rgb_array(self):
        """Test loading RGB numpy array."""
        # Create RGB array (red image)
        rgb_array = np.zeros((80, 120, 3), dtype=np.uint8)
        rgb_array[:, :, 0] = 255  # Red channel in RGB

        loader = ImageLoader()
        loaded = loader.load_from_array(rgb_array, color_space="RGB")

        assert loaded.image.shape == (80, 120, 3)
        assert loaded.orientation == 1

        # Red in RGB should become blue in BGR
        assert loaded.image[0, 0, 0] == 0  # Blue channel
        assert loaded.image[0, 0, 2] == 255  # Red channel

    def test_load_gray_array(self):
        """Test loading grayscale numpy array."""
        # Create grayscale array
        gray_array = np.full((90, 110), 150, dtype=np.uint8)

        loader = ImageLoader()
        loaded = loader.load_from_array(gray_array, color_space="GRAY")

        assert loaded.image.shape == (90, 110, 3)
        assert loaded.orientation == 1

        # All channels should be equal
        assert np.allclose(loaded.image[0, 0], [150, 150, 150])

    def test_invalid_color_space(self):
        """Test error handling for invalid color space."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)

        loader = ImageLoader()
        with pytest.raises(ValueError, match="Unsupported color_space"):
            loader.load_from_array(array, color_space="INVALID")

    def test_invalid_array_dimensions(self):
        """Test error handling for invalid array dimensions."""
        # 1D array is invalid
        array = np.zeros(100, dtype=np.uint8)

        loader = ImageLoader()
        with pytest.raises(ValueError, match="Expected 2D or 3D array"):
            loader.load_from_array(array, color_space="BGR")

    def test_load_3d_gray_array(self):
        """Test loading 3D grayscale array (single channel)."""
        # Create 3D grayscale array
        gray_array = np.full((100, 100, 1), 180, dtype=np.uint8)

        loader = ImageLoader()
        loaded = loader.load_from_array(gray_array, color_space="GRAY")

        assert loaded.image.shape == (100, 100, 3)
        assert np.allclose(loaded.image[0, 0], [180, 180, 180])


class TestImageLoaderEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_very_small_image(self, tmp_path: Path):
        """Test loading very small image (1x1 pixel)."""
        img = Image.new("RGB", (1, 1), color=(100, 200, 50))
        img_path = tmp_path / "tiny.png"
        img.save(img_path)

        loader = ImageLoader()
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.image.shape == (1, 1, 3)
        assert loaded.canonical_width == 1
        assert loaded.canonical_height == 1

    def test_large_aspect_ratio_image(self, tmp_path: Path):
        """Test loading image with extreme aspect ratio."""
        img = Image.new("RGB", (1000, 10), color=(255, 255, 255))
        img_path = tmp_path / "wide.png"
        img.save(img_path)

        loader = ImageLoader()
        loaded = loader.load_from_path(img_path)

        assert loaded is not None
        assert loaded.image.shape == (10, 1000, 3)
        assert loaded.canonical_width == 1000
        assert loaded.canonical_height == 10

    def test_corrupted_image_file(self, tmp_path: Path):
        """Test handling of corrupted image file."""
        # Create invalid image file
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"not a valid image")

        loader = ImageLoader()
        loaded = loader.load_from_path(corrupt_path)

        # Should return None for corrupted files
        assert loaded is None

    def test_empty_file(self, tmp_path: Path):
        """Test handling of empty file."""
        empty_path = tmp_path / "empty.png"
        empty_path.touch()

        loader = ImageLoader()
        loaded = loader.load_from_path(empty_path)

        assert loaded is None


class TestLoadedImageDataclass:
    """Tests for LoadedImage dataclass."""

    def test_loaded_image_attributes(self):
        """Test LoadedImage dataclass attributes."""
        image = np.zeros((100, 150, 3), dtype=np.uint8)

        loaded = LoadedImage(
            image=image,
            orientation=6,
            raw_width=150,
            raw_height=100,
            canonical_width=100,
            canonical_height=150,
        )

        assert loaded.image.shape == (100, 150, 3)
        assert loaded.orientation == 6
        assert loaded.raw_width == 150
        assert loaded.raw_height == 100
        assert loaded.canonical_width == 100
        assert loaded.canonical_height == 150

    def test_loaded_image_is_dataclass(self):
        """Test that LoadedImage is a dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(LoadedImage)
