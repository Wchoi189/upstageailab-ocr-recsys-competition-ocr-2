"""
Tests for PLM data contracts and type safety.
"""

import pytest
import torch

from ocr.core.interfaces.plm import AttentionMasks, PLMConfig, PLMLossConfig


class TestPLMConfig:
    """Test PLMConfig dataclass validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = PLMConfig(max_len=25, perm_num=6, perm_forward=True, perm_mirrored=True)
        assert config.max_len == 25
        assert config.perm_num == 6
        assert config.max_gen_perms == 3

    def test_invalid_max_len(self):
        """Test max_len must be positive."""
        with pytest.raises(ValueError, match="max_len must be positive"):
            PLMConfig(max_len=0)

    def test_mirrored_requires_even(self):
        """Test perm_mirrored requires even perm_num."""
        with pytest.raises(ValueError, match="perm_num must be even"):
            PLMConfig(perm_num=5, perm_mirrored=True)


class TestAttentionMasks:
    """Test AttentionMasks dataclass validation."""

    def test_valid_masks(self):
        """Test valid attention masks."""
        content = torch.ones(4, 4, dtype=torch.bool)
        query = torch.ones(4, 4, dtype=torch.bool)
        masks = AttentionMasks(content_mask=content, query_mask=query)

        assert masks.sequence_length == 4

    def test_shape_mismatch(self):
        """Test masks must have same shape."""
        content = torch.ones(4, 4, dtype=torch.bool)
        query = torch.ones(5, 5, dtype=torch.bool)

        with pytest.raises(ValueError, match="Mask shapes must match"):
            AttentionMasks(content_mask=content, query_mask=query)
