"""
*****************************************************************************************
* Modified from https://github.com/MhLiao/DB/blob/master/decoders/balance_cross_entropy_loss.py
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

from ocr.core.validation import ValidatedTensorData


class BCELoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6, validate_inputs=False):
        """BCE Loss with OHEM (Online Hard Example Mining).

        Args:
            negative_ratio: Ratio of negative to positive samples for OHEM
            eps: Small epsilon for numerical stability
            validate_inputs: Enable Pydantic validation (SLOW - debug only)
        """
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps
        self.validate_inputs = validate_inputs

    def forward(self, pred_logits, gt, mask=None):
        if mask is None:
            mask = torch.ones_like(gt, device=gt.device, dtype=gt.dtype)

        # Pydantic validation (debug mode only - causes GPU sync)
        if self.validate_inputs:
            try:
                ValidatedTensorData(
                    tensor=pred_logits,
                    expected_shape=tuple(pred_logits.shape),
                    expected_device=pred_logits.device,
                    allow_nan=False,
                    allow_inf=False,
                )
                ValidatedTensorData(
                    tensor=gt,
                    expected_shape=tuple(gt.shape),
                    expected_device=gt.device,
                    value_range=(0.0, 1.0),
                    allow_nan=False,
                    allow_inf=False,
                )
                ValidatedTensorData(
                    tensor=mask, expected_shape=tuple(mask.shape), expected_device=mask.device, allow_nan=False, allow_inf=False
                )
            except ValidationError as exc:
                raise ValueError(f"BCE loss input validation failed: {exc}") from exc

        positive = (gt * mask) > 0
        negative = ((1 - gt) * mask) > 0

        # Single GPU sync for OHEM topk (unavoidable for topk k parameter)
        positive_count = positive.sum()
        neg_sum = negative.sum()
        max_neg = (positive_count * self.negative_ratio).int()
        negative_count = torch.minimum(neg_sum, max_neg).item()
        positive_count = positive_count.item()

        loss = nn.functional.binary_cross_entropy_with_logits(pred_logits, gt, reduction="none")

        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        if negative_count > 0:
            negative_loss, _ = torch.topk(negative_loss.view(-1), int(negative_count))
            negative_loss_sum = negative_loss.sum()
        else:
            negative_loss_sum = torch.zeros((), device=loss.device, dtype=loss.dtype)

        balance_loss = (positive_loss.sum() + negative_loss_sum) / (positive_count + negative_count + self.eps)

        return balance_loss
