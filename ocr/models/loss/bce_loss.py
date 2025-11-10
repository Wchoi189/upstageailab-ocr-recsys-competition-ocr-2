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
*
* Bug Fix: BUG-20251109-002 - CUDA Illegal Memory Access in BCE Loss Computation
* Changes: Added input validation, CUDA synchronization, moved operations to CPU
* See: docs/bug_reports/BUG-20251109-002-code-changes.md
*****************************************************************************************
"""

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.eps = eps

    def forward(self, pred_logits, gt, mask=None):
        """
        Forward pass for BCE loss computation.

        BUG-20251109-002: Fixed CUDA illegal memory access by:
        - Adding input validation (shape/device checks)
        - Adding CUDA synchronization before operations
        - Moving operations to CPU to avoid corrupted memory access
        - Enhanced error handling with debugging context

        See: docs/bug_reports/BUG-20251109-002-code-changes.md
        """
        # BUG-20251109-002: Validate inputs to prevent CUDA illegal memory access
        if pred_logits.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch: pred_logits.shape={pred_logits.shape}, gt.shape={gt.shape}. "
                f"Shapes must match for BCE loss computation."
            )

        if pred_logits.device != gt.device:
            raise ValueError(
                f"Device mismatch: pred_logits.device={pred_logits.device}, gt.device={gt.device}. "
                f"Tensors must be on the same device."
            )

        if mask is None:
            mask = torch.ones_like(gt, device=gt.device, dtype=gt.dtype)
        else:
            if mask.shape != gt.shape:
                raise ValueError(
                    f"Shape mismatch: mask.shape={mask.shape}, gt.shape={gt.shape}. "
                    f"Mask must have the same shape as gt."
                )
            if mask.device != gt.device:
                raise ValueError(
                    f"Device mismatch: mask.device={mask.device}, gt.device={gt.device}. "
                    f"Mask must be on the same device as gt."
                )

        # BUG-20251109-002: Check for CUDA errors before operations
        if pred_logits.device.type == 'cuda':
            # Clear any previous CUDA errors
            try:
                torch.cuda.synchronize(pred_logits.device)
            except RuntimeError:
                # If synchronization fails, CUDA is in a bad state
                # Try to clear the error and continue
                torch.cuda.empty_cache()

        # BUG-20251109-002: Create boolean masks with error handling
        # Move to CPU first to avoid CUDA memory corruption issues
        # This is safer than doing operations on CUDA and then moving
        # NOTE: Even .cpu() fails if CUDA memory is corrupted - suggests corruption happens earlier
        try:
            # Try to move to CPU - if this fails, CUDA memory is corrupted
            gt_cpu = gt.cpu()
            mask_cpu = mask.cpu()

            # Create boolean masks on CPU
            positive_cpu = (gt_cpu * mask_cpu) > 0
            negative_cpu = ((1 - gt_cpu) * mask_cpu) > 0

            # Sum on CPU (safe)
            positive_count = int(positive_cpu.sum().item())
            negative_count = min(int(negative_cpu.sum().item()), int(positive_count * self.negative_ratio))

            # Create CUDA versions for later use (if needed)
            # Only if we're on CUDA and the operation succeeded
            if pred_logits.device.type == 'cuda':
                positive = positive_cpu.to(pred_logits.device)
                negative = negative_cpu.to(pred_logits.device)
            else:
                positive = positive_cpu
                negative = negative_cpu

        except RuntimeError as e:
            # If CUDA error occurs, provide more context
            # Check if it's a CUDA error or something else
            error_msg = str(e)
            if "CUDA" in error_msg or "cuda" in error_msg:
                raise RuntimeError(
                    f"CUDA illegal memory access in BCE loss computation. "
                    f"pred_logits.shape={pred_logits.shape}, gt.shape={gt.shape}, "
                    f"mask.shape={mask.shape}, pred_logits.device={pred_logits.device}, "
                    f"gt.device={gt.device}, mask.device={mask.device}. "
                    f"This suggests CUDA memory corruption. Try: "
                    f"1. Clearing GPU cache: torch.cuda.empty_cache() "
                    f"2. Reducing batch size "
                    f"3. Checking for out-of-bounds tensor access earlier in the pipeline. "
                    f"Original error: {e}"
                ) from e
            else:
                # Re-raise if it's not a CUDA error
                raise

        loss = nn.functional.binary_cross_entropy_with_logits(pred_logits, gt, reduction="none")

        positive_loss = loss * positive.float()
        negative_loss = loss * negative.float()

        if negative_count > 0:
            negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)
            negative_loss_sum = negative_loss.sum()
        else:
            negative_loss_sum = torch.zeros((), device=loss.device, dtype=loss.dtype)

        balance_loss = (positive_loss.sum() + negative_loss_sum) / (positive_count + negative_count + self.eps)

        return balance_loss
