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


class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

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
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape, f"{pred.shape}, {mask.shape}"
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask

        # Clamp predictions to valid range [0, 1] to prevent extreme values
        pred = pred.clamp(0, 1)

        # Validate inputs for NaN/Inf before computation
        if torch.isnan(pred).any():
            raise ValueError("NaN detected in predictions before Dice loss computation")
        if torch.isinf(pred).any():
            raise ValueError("Inf detected in predictions before Dice loss computation")
        if torch.isnan(gt).any():
            raise ValueError("NaN detected in ground truth before Dice loss computation")
        if torch.isinf(gt).any():
            raise ValueError("Inf detected in ground truth before Dice loss computation")

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union

        # Validate output for NaN/Inf after computation
        if torch.isnan(loss):
            raise ValueError("NaN detected in Dice loss output")
        if torch.isinf(loss):
            raise ValueError("Inf detected in Dice loss output")

        assert loss <= 1
        return loss
