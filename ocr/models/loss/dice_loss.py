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

        # BUG-20251110-002: Enhanced input validation to catch NaN/Inf before computation
        if torch.isnan(pred).any():
            raise ValueError(
                f"NaN values in pred input to DiceLoss. "
                f"Shape: {pred.shape}, Range: [{pred.min().item():.6f}, {pred.max().item():.6f}]"
            )
        if torch.isinf(pred).any():
            raise ValueError(
                f"Inf values in pred input to DiceLoss. "
                f"Shape: {pred.shape}, Range: [{pred.min().item():.6f}, {pred.max().item():.6f}]"
            )

        # Clamp predictions to [0, 1] to ensure numerical stability
        # This prevents loss from exceeding 1 due to values outside expected range
        pred = pred.clamp(0, 1)

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps

        # BUG-20251110-002: Check for degenerate cases before division
        if union < self.eps * 2:  # Union should be at least 2*eps
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Degenerate case in DiceLoss: union={union.item():.6e} < 2*eps. "
                f"This can cause numerical instability. Setting loss to 1.0 (worst case)."
            )
            return torch.tensor(1.0, device=pred.device, dtype=pred.dtype)

        loss = 1 - 2.0 * intersection / union

        # BUG-20251110-002: Validate loss value before returning
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"NaN/Inf loss computed in DiceLoss. "
                f"Intersection: {intersection.item():.6e}, Union: {union.item():.6e}, "
                f"Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}], "
                f"GT range: [{gt.min().item():.4f}, {gt.max().item():.4f}]"
            )

        # Allow small numerical errors (e.g., 1e-5) but warn if significantly > 1
        if loss > 1.01:  # More lenient check with small tolerance
            import warnings
            warnings.warn(
                f"Dice loss exceeds 1: {loss.item():.6f}. "
                f"Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}], "
                f"GT range: [{gt.min().item():.4f}, {gt.max().item():.4f}], "
                f"Intersection: {intersection.item():.6f}, Union: {union.item():.6f}",
                RuntimeWarning
            )
            # Clamp loss to reasonable range [0, 2] to prevent training crash
            loss = loss.clamp(0, 2)

        return loss
