"""Image Preprocessing Module.

Handles image resizing, format conversion, and optimization for VLM analysis.
"""

import base64
import io
from pathlib import Path

from PIL import Image

from AgentQMS.vlm.core.config import get_config
from AgentQMS.vlm.core.contracts import ImageFormat, ProcessedImage
from AgentQMS.vlm.core.interfaces import ImagePreprocessor, PreprocessingError


class VLMImagePreprocessor(ImagePreprocessor):
    """Image preprocessor for VLM analysis."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize preprocessor.

        Args:
            cache_dir: Optional directory to cache processed images
        """
        self.cache_dir = cache_dir
        self._image_config = get_config().image
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def preprocess(
        self,
        image_path: Path,
        max_resolution: int,
        target_format: str | None = None,
    ) -> ProcessedImage:
        """Preprocess an image for VLM analysis.

        Args:
            image_path: Path to input image
            max_resolution: Maximum resolution (width or height)
            target_format: Target image format (JPEG, PNG, etc.)

        Returns:
            Processed image data

        Raises:
            PreprocessingError: If preprocessing fails
        """
        try:
            # Load image
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                original_format = img.format or "JPEG"

                # Convert to RGB if necessary (for JPEG compatibility)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Calculate resize dimensions
                width, height, resize_ratio = self._calculate_resize(original_width, original_height, max_resolution)

                # Resize if needed
                if resize_ratio < 1.0:
                    img = img.resize((width, height), Image.Resampling.LANCZOS)

                # Determine target format
                if target_format is None:
                    target_format = original_format
                target_format = target_format.upper()

                # Validate format
                try:
                    format_enum = ImageFormat(target_format)
                except ValueError:
                    # Default to JPEG if format not supported
                    format_enum = ImageFormat.JPEG
                    target_format = "JPEG"

                # Save to bytes
                output = io.BytesIO()
                save_format = format_enum.value
                img.save(
                    output,
                    format=save_format,
                    quality=self._image_config.default_quality,
                    optimize=True,
                )
                output.seek(0)
                processed_bytes = output.read()
                size_bytes = len(processed_bytes)

                # Encode to base64
                base64_encoded = base64.b64encode(processed_bytes).decode("utf-8")

                # Optionally save to cache
                processed_path = None
                if self.cache_dir:
                    cache_filename = f"{image_path.stem}_processed_{width}x{height}.{save_format.lower()}"
                    processed_path = self.cache_dir / cache_filename
                    processed_path.write_bytes(processed_bytes)

                return ProcessedImage(
                    original_path=image_path,
                    processed_path=processed_path,
                    format=format_enum,
                    width=width,
                    height=height,
                    original_width=original_width,
                    original_height=original_height,
                    resize_ratio=resize_ratio,
                    base64_encoded=base64_encoded,
                    size_bytes=size_bytes,
                    metadata={
                        "original_format": original_format,
                        "target_format": target_format,
                        "resized": resize_ratio < 1.0,
                    },
                )

        except Exception as e:
            raise PreprocessingError(f"Failed to preprocess image {image_path}: {e}") from e

    def preprocess_batch(
        self,
        image_paths: list[Path],
        max_resolution: int,
        target_format: str | None = None,
    ) -> list[ProcessedImage]:
        """Preprocess multiple images.

        Args:
            image_paths: List of image paths
            max_resolution: Maximum resolution (width or height)
            target_format: Target image format

        Returns:
            List of processed image data
        """
        results = []
        for image_path in image_paths:
            try:
                processed = self.preprocess(image_path, max_resolution, target_format)
                results.append(processed)
            except PreprocessingError as e:
                # Log error but continue with other images
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Skipping image {image_path} due to preprocessing error: {e}")
                continue

        return results

    @staticmethod
    def _calculate_resize(original_width: int, original_height: int, max_resolution: int) -> tuple[int, int, float]:
        """Calculate resize dimensions preserving aspect ratio.

        Args:
            original_width: Original image width
            original_height: Original image height
            max_resolution: Maximum resolution (width or height)

        Returns:
            Tuple of (new_width, new_height, resize_ratio)
        """
        if original_width <= max_resolution and original_height <= max_resolution:
            # No resize needed
            return original_width, original_height, 1.0

        # Calculate resize ratio
        width_ratio = max_resolution / original_width
        height_ratio = max_resolution / original_height
        resize_ratio = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(original_width * resize_ratio)
        new_height = int(original_height * resize_ratio)

        # Ensure dimensions are even (some models prefer this)
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)

        return new_width, new_height, resize_ratio
