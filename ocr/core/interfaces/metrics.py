"""Abstract base classes for OCR evaluation metrics."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseMetric(nn.Module, ABC):
    """Abstract base class for OCR evaluation metrics.

    Metrics compute evaluation scores for text detection performance.
    """

    def __init__(self, **kwargs):
        """Initialize the metric.

        Args:
            **kwargs: Configuration parameters for the metric.
        """
        super().__init__()

    @abstractmethod
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric state with new predictions and targets.

        Args:
            preds: Predicted polygons or maps
            targets: Ground truth polygons or maps
        """
        pass

    @abstractmethod
    def compute(self) -> dict[str, torch.Tensor]:
        """Compute final metric values.

        Returns:
            Dictionary of metric values (precision, recall, f1, etc.)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset metric state for new evaluation."""
        pass
