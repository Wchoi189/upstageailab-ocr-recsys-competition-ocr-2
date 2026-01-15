"""Abstract base classes for OCR loss functions."""

from abc import ABC, abstractmethod
from typing import Any
import torch
import torch.nn as nn


class BaseLoss(nn.Module, ABC):
    """Abstract base class for OCR loss functions.

    Loss functions compute training objectives for text detection models.
    """

    def __init__(self, **kwargs):
        """Initialize the loss function.

        Args:
            **kwargs: Configuration parameters for the loss function.
        """
        super().__init__()

    @abstractmethod
    def forward(
        self, preds: Any, targets: Any, **kwargs
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss from predictions and ground truth.

        Args:
            preds: Model predictions (usually a tensor or dict)
            targets: Ground truth targets (tensor or dict)
            **kwargs: Additional loss-specific parameters

        Returns:
            Tuple of (total_loss, loss_dict) where:
            - total_loss: Scalar tensor for optimization
            - loss_dict: Dictionary of individual loss components for logging
        """
        pass
