"""Interfaces for detection feature components."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from ocr.core.interfaces.losses import BaseLoss
from ocr.core.interfaces.models import BaseHead


class DetectionHead(BaseHead, ABC):
    """Abstract base class for Detection prediction heads."""

    @abstractmethod
    def get_polygons_from_maps(
        self,
        batch: dict[str, Any],
        pred: dict[str, torch.Tensor]
    ) -> tuple[list[list[list[int]]], list[list[float]]]:
        """Extract polygons and scores from prediction maps.

        Args:
            batch: Batch dictionary with preprocessing metadata including:
                - images: Tensor of shape (B, C, H, W)
                - shape: List of original image dimensions before preprocessing
                - filename: List of source image filenames
                - inverse_matrix: Matrices for mapping predictions back to original coords
            pred: Dictionary of prediction maps from model forward pass.
                  Keys depend on head type (e.g., 'binary_map', 'thresh_map' for DB)

        Returns:
            Tuple containing:
            - boxes_batch: List[List[List[int]]] - Polygons per image with integer coordinates
                          Shape: [batch_size][num_boxes][num_points*2]
                          Each box is flattened [x1,y1,x2,y2,...,xn,yn]
            - scores_batch: List[List[float]] - Confidence scores per box
                           Shape: [batch_size][num_boxes]
        """
        pass


class DetectionLoss(BaseLoss, ABC):
    """Abstract base class for Detection loss functions."""

    @abstractmethod
    def forward(
        self, pred: dict[str, torch.Tensor], gt_binary: torch.Tensor, gt_thresh: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss from predictions and ground truth.

        Args:
            pred: Dictionary of predictions from head
            gt_binary: Ground truth binary map of shape (B, 1, H, W)
            gt_thresh: Ground truth threshold map of shape (B, 1, H, W)
            **kwargs: Additional loss-specific parameters

        Returns:
            Tuple of (total_loss, loss_dict) where:
            - total_loss: Scalar tensor for optimization
            - loss_dict: Dictionary of individual loss components for logging
        """
        pass
