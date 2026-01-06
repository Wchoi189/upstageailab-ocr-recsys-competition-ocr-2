"""Unit tests for CrossEntropyLoss."""

import pytest
import torch


class TestCrossEntropyLoss:
    """Test CrossEntropyLoss for recognition tasks."""

    @pytest.fixture
    def loss_fn(self):
        """Create a CrossEntropyLoss instance."""
        from ocr.models.loss.cross_entropy_loss import CrossEntropyLoss

        return CrossEntropyLoss(ignore_index=0)

    def test_forward_basic(self, loss_fn):
        """Test basic forward pass with valid inputs."""
        B, T, V = 2, 10, 100  # batch, time, vocab
        logits = torch.randn(B, T, V)
        targets = torch.randint(1, V, (B, T))  # Avoid 0 (PAD)

        loss, loss_dict = loss_fn(logits, targets)

        assert loss.shape == ()  # Scalar
        assert "loss_ce" in loss_dict
        assert loss_dict["loss_ce"] == loss

    def test_ignore_index_masks_padding(self, loss_fn):
        """Test that ignore_index correctly masks padding tokens."""
        B, T, V = 2, 10, 100
        logits = torch.randn(B, T, V)

        # Create targets with lots of padding (0)
        targets = torch.zeros(B, T, dtype=torch.long)
        targets[:, :3] = torch.randint(1, V, (B, 3))  # Only first 3 are real

        loss, _ = loss_fn(logits, targets)

        # Loss should be finite (not NaN/Inf due to masking)
        assert torch.isfinite(loss)

    def test_label_smoothing(self):
        """Test that label smoothing is applied when specified."""
        from ocr.models.loss.cross_entropy_loss import CrossEntropyLoss

        loss_no_smooth = CrossEntropyLoss(ignore_index=0, label_smoothing=0.0)
        loss_with_smooth = CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

        B, T, V = 2, 5, 50
        logits = torch.randn(B, T, V)
        targets = torch.randint(1, V, (B, T))

        loss1, _ = loss_no_smooth(logits, targets)
        loss2, _ = loss_with_smooth(logits, targets)

        # With same inputs, smoothing typically increases loss slightly
        assert loss1 != loss2

    def test_loss_registered_in_registry(self):
        """Test that cross_entropy is registered in the component registry."""
        from ocr.models.core import registry

        losses = registry.list_losses()
        assert "cross_entropy" in losses
