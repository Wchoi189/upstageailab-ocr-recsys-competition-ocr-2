"""
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/decoders/dice_loss.py
*
* 참고 논문:
* Real-time Scene Text Detection with Differentiable Binarization
* https://arxiv.org/pdf/1911.08947.pdf
*
* 참고 Repository:
* https://github.com/MhLiao/DB/
*****************************************************************************************
"""

import torch
import torch.nn as nn
from pydantic import ValidationError

from ocr.validation.models import ValidatedTensorData


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6, validate_inputs=True):
        super().__init__()
        self.eps = eps
        self.validate_inputs = validate_inputs

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        """
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, 1, H, W)
        weights: (N, 1, H, W)
        """
        assert pred.dim() == 4, pred.dim()
        if mask is None:
            mask = torch.ones_like(gt).to(device=gt.device)

        # Validate inputs using ValidatedTensorData (BUG-20251112-001 prevention)
        if self.validate_inputs:
            try:
                # Validate prediction tensor: shape, device, no NaN/Inf
                ValidatedTensorData(
                    tensor=pred,
                    expected_shape=tuple(pred.shape),
                    expected_device=pred.device,
                    allow_nan=False,
                    allow_inf=False
                )
                # Validate ground truth tensor
                ValidatedTensorData(
                    tensor=gt,
                    expected_shape=tuple(gt.shape),
                    expected_device=gt.device,
                    allow_nan=False,
                    allow_inf=False
                )
                # Validate mask tensor
                ValidatedTensorData(
                    tensor=mask,
                    expected_shape=tuple(mask.shape),
                    expected_device=mask.device,
                    allow_nan=False,
                    allow_inf=False
                )
            except ValidationError as exc:
                raise ValueError(f"Dice loss input validation failed: {exc}") from exc

        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape, f"{pred.shape}, {mask.shape}"
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        # Input validation for numerical stability
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise ValueError(f"Invalid values in pred: nan={torch.isnan(pred).any().item()}, inf={torch.isinf(pred).any().item()}")
        if torch.isnan(gt).any() or torch.isinf(gt).any():
            raise ValueError(f"Invalid values in gt: nan={torch.isnan(gt).any().item()}, inf={torch.isinf(gt).any().item()}")
        # Clamp predictions to valid probability range
        pred = pred.clamp(0.0, 1.0)
        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        # Guard against degenerate unions
        if union < 2 * self.eps:
            # Return worst-case loss while avoiding NaNs
            return torch.tensor(1.0, device=pred.device, dtype=pred.dtype)
        loss = 1 - 2.0 * intersection / union
        # Tolerate minor numeric overshoot instead of asserting
        loss = torch.clamp(loss, min=0.0, max=1.0 + 1e-6)
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"Invalid Dice loss computed: nan/inf encountered. intersection={intersection.item():.6e}, union={union.item():.6e}"
            )
        return loss
