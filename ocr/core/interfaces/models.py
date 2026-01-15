"""Abstract base classes for OCR model components."""

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for OCR encoders/backbones.

    Encoders are responsible for extracting features from input images.
    They should return a list of feature maps at different scales.
    """

    def __init__(self, **kwargs):
        """Initialize the encoder.

        Args:
            **kwargs: Configuration parameters specific to the encoder implementation.
        """
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature tensors at different scales, typically from shallow to deep layers.
            Each tensor should have shape (B, C_i, H_i, W_i) where H_i, W_i decrease with depth.
        """
        pass

    @property
    @abstractmethod
    def out_channels(self) -> list[int]:
        """Return the number of output channels for each feature level.

        Returns:
            List of integers representing output channels for each feature level.
        """
        pass

    @property
    @abstractmethod
    def strides(self) -> list[int]:
        """Return the stride (downsampling factor) for each feature level.

        Returns:
            List of integers representing stride for each feature level.
        """
        pass

    def get_feature_info(self) -> dict[str, Any]:
        """Get information about the features produced by this encoder.

        Returns:
            Dictionary containing feature information like channels, strides, etc.
        """
        return {
            "out_channels": self.out_channels,
            "strides": self.strides,
            "num_levels": len(self.out_channels),
        }


class BaseDecoder(nn.Module, ABC):
    """Abstract base class for OCR decoders.

    Decoders take encoder features and produce higher-resolution feature maps
    suitable for dense prediction tasks like text detection.
    """

    def __init__(self, in_channels: list[int], **kwargs):
        """Initialize the decoder.

        Args:
            in_channels: List of input channels from encoder feature levels.
            **kwargs: Additional configuration parameters.
        """
        super().__init__()
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, features: list[torch.Tensor], targets: torch.Tensor = None) -> torch.Tensor:
        """Decode features to produce dense prediction maps.

        Args:
            features: List of feature tensors from encoder, ordered from shallow to deep.
            targets: Optional targets for autoregressive decoding (e.g. text tokens).

        Returns:
            Decoded feature tensor suitable for head prediction.
            Typically shape (B, C, H, W) where H, W match input image dimensions.
        """
        pass

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Return the number of output channels from the decoder.

        Returns:
            Number of output channels.
        """
        pass


class BaseHead(nn.Module, ABC):
    """Abstract base class for OCR prediction heads.

    Heads take decoded features and produce final predictions like
    text region segmentation maps or bounding boxes.
    """

    def __init__(self, in_channels: int, **kwargs):
        """Initialize the head.

        Args:
            in_channels: Number of input channels from decoder.
            **kwargs: Additional configuration parameters.
        """
        super().__init__()
        self.in_channels = in_channels

    @abstractmethod
    def forward(self, x: torch.Tensor, return_loss: bool = True) -> dict[str, torch.Tensor]:
        """Produce predictions from decoded features.

        Args:
            x: Input tensor from decoder of shape (B, C, H, W)
            return_loss: Whether to include loss computation in output

        Returns:
            Dictionary containing predictions and optionally loss.
            Must include keys like 'binary_map', 'thresh_map', etc.
        """
        pass


