"""Background removal transform using rembg AI models.

This module provides an Albumentations-compatible transform for removing
backgrounds from images using the rembg library with ONNX models.
"""

from __future__ import annotations

import numpy as np
from albumentations import ImageOnlyTransform

try:
    from rembg import remove

    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False


class BackgroundRemoval(ImageOnlyTransform):
    """Remove background using rembg AI model.

    This transform uses the rembg library to automatically remove backgrounds
    from images using deep learning models. The result is composited on a
    white background for compatibility with OCR pipelines.

    Args:
        model: Model name to use for background removal.
            Options: "u2net" (default, best quality), "u2netp" (faster),
                    "u2net_cloth_seg", "silueta", "isnet"
        alpha_matting: Enable alpha matting for better edge quality
        alpha_matting_foreground_threshold: Foreground threshold for alpha matting
        alpha_matting_background_threshold: Background threshold for alpha matting
        alpha_matting_erode_size: Erosion size for alpha matting
        only_mask: Return only the mask instead of the processed image
        post_process_mask: Apply post-processing to the mask
        always_apply: Always apply the transform (Albumentations parameter)
        p: Probability of applying the transform (Albumentations parameter)

    Example:
        >>> import cv2
        >>> from ocr.data.datasets.preprocessing.background_removal import BackgroundRemoval
        >>>
        >>> # Create transform
        >>> bg_remover = BackgroundRemoval(model="u2net", alpha_matting=True, p=1.0)
        >>>
        >>> # Apply to image
        >>> image = cv2.imread("image.jpg")
        >>> result = bg_remover(image=image)["image"]
        >>> cv2.imwrite("output.png", result)

    Note:
        - Models are downloaded automatically on first use (~176MB for u2net)
        - GPU acceleration is automatic if CUDA/ONNX Runtime GPU is available
        - Results are composited on white background for OCR compatibility
    """

    def __init__(
        self,
        model: str = "u2net",
        alpha_matting: bool = True,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        only_mask: bool = False,
        post_process_mask: bool = False,
        always_apply: bool = False,
        p: float = 1.0,
    ):
        super().__init__(always_apply=always_apply, p=p)

        if not REMBG_AVAILABLE:
            raise ImportError("rembg is not installed. Install it with: pip install rembg onnxruntime")

        self.model = model
        self.alpha_matting = alpha_matting
        self.alpha_matting_foreground_threshold = alpha_matting_foreground_threshold
        self.alpha_matting_background_threshold = alpha_matting_background_threshold
        self.alpha_matting_erode_size = alpha_matting_erode_size
        self.only_mask = only_mask
        self.post_process_mask = post_process_mask

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        """Apply background removal to the image.

        Args:
            img: Input image as numpy array (RGB or BGR format)
            **params: Additional parameters (unused)

        Returns:
            Image with background removed, composited on white background
        """
        # Remove background using rembg
        output = remove(
            img,
            alpha_matting=self.alpha_matting,
            alpha_matting_foreground_threshold=self.alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=self.alpha_matting_background_threshold,
            alpha_matting_erode_size=self.alpha_matting_erode_size,
            only_mask=self.only_mask,
            post_process_mask=self.post_process_mask,
        )

        # If only mask requested, return as-is
        if self.only_mask:
            return output

        # Composite on white background (for OCR compatibility)
        if output.shape[2] == 4:  # RGBA format
            rgb = output[:, :, :3]
            alpha = output[:, :, 3:4] / 255.0
            white_bg = np.ones_like(rgb) * 255
            result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            return result

        # Already RGB, return as-is
        return output

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        """Get names of arguments used in __init__.

        Required by Albumentations for serialization.
        """
        return (
            "model",
            "alpha_matting",
            "alpha_matting_foreground_threshold",
            "alpha_matting_background_threshold",
            "alpha_matting_erode_size",
            "only_mask",
            "post_process_mask",
        )


def create_background_removal_transform(
    model: str = "u2net",
    alpha_matting: bool = True,
    p: float = 1.0,
) -> BackgroundRemoval:
    """Convenience function to create a background removal transform.

    Args:
        model: Model name ("u2net", "u2netp", etc.)
        alpha_matting: Enable alpha matting for better edges
        p: Probability of applying the transform

    Returns:
        Configured BackgroundRemoval transform

    Example:
        >>> bg_remover = create_background_removal_transform(model="u2net", p=1.0)
        >>> result = bg_remover(image=image)["image"]
    """
    return BackgroundRemoval(
        model=model,
        alpha_matting=alpha_matting,
        p=p,
    )
