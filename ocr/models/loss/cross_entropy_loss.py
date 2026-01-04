"""Cross-Entropy Loss for text recognition (PARSeq, etc.).

BUG_003: Created as part of PARSeq component implementation.
"""

import torch
import torch.nn.functional as F

from ..core import BaseLoss


class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss for sequence-to-sequence recognition tasks.

    Unlike detection losses (DBLoss, etc.) that operate on spatial maps,
    this loss works with token logits for autoregressive text recognition.

    Args:
        ignore_index: Token index to ignore in loss computation (typically PAD=0).
        label_smoothing: Label smoothing coefficient (0.0 = no smoothing).
    """

    def __init__(
        self,
        ignore_index: int = 0,
        label_smoothing: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute cross-entropy loss for recognition.

        Args:
            logits: Predicted logits of shape [B, T, V] (batch, time, vocab).
            targets: Target token indices of shape [B, T].
            **kwargs: Additional arguments (ignored for compatibility).

        Returns:
            Tuple of (total_loss, loss_dict) where:
            - total_loss: Scalar tensor for optimization.
            - loss_dict: Dictionary with {"loss_ce": total_loss} for logging.
        """
        # Reshape for cross_entropy: [B, T, V] -> [B*T, V], [B, T] -> [B*T]
        B, T, V = logits.shape
        logits_flat = logits.reshape(B * T, V)
        targets_flat = targets.reshape(B * T)

        loss = F.cross_entropy(
            logits_flat,
            targets_flat,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )

        return loss, {"loss_ce": loss}
