"""
Permutation Language Modeling (PLM) Module

This module implements the core PLM logic extracted from parseq_official_adapter.py.
PLM is the training strategy that generates K random permutations of token orderings.

CRITICAL: This is an EXACT extraction from the working reference implementation.
Do NOT modify the logic - numerical equivalence is required.

References:
    - Source: ocr/domains/recognition/models/parseq_official_adapter.py:92-156
    - Contracts: ocr/core/interfaces/plm.py
    - Paper: https://arxiv.org/abs/2207.06966
"""

import math
from itertools import permutations

import numpy as np
import torch
from torch import Tensor

from ocr.core.interfaces.plm import AttentionMasks, PLMConfig


class PermutationLanguageModeling:
    """
    Permutation Language Modeling implementation.

    This class generates K permutations for target sequences and creates
    attention masks that enforce causal structure based on permutation ordering.

    Args:
        max_label_length: Maximum sequence length (excluding BOS/EOS)
        perm_num: Number of permutations to generate (K in paper)
        perm_forward: Whether to include forward (left-to-right) permutation
        perm_mirrored: Whether to generate mirrored (complementary) pairs
        device: Device for tensor operations (cuda/cpu)
        seed: Random seed for reproducibility

    Example:
        >>> plm = PermutationLanguageModeling(max_label_length=25)
        >>> tgt = torch.tensor([[1, 5, 10, 7, 2]])  # [B, L]
        >>> perms = plm.gen_tgt_perms(tgt)  # [K, L]
        >>> for perm in perms:
        ...     masks = plm.generate_attn_masks(perm)
        ...     # Use masks in decoder
    """

    def __init__(
        self,
        max_label_length: int = 25,
        perm_num: int = 6,
        perm_forward: bool = True,
        perm_mirrored: bool = True,
        device: str = "cuda",
        seed: int = 42,
    ):
        # Validate using config
        self.config = PLMConfig(
            max_len=max_label_length,
            perm_num=perm_num,
            perm_forward=perm_forward,
            perm_mirrored=perm_mirrored,
        )

        # Store attributes
        self.max_label_length = max_label_length
        self.perm_num = perm_num
        self.perm_forward = perm_forward
        self.perm_mirrored = perm_mirrored
        self.max_gen_perms = self.config.max_gen_perms
        self._device = device

        # Initialize RNG for permutation sampling
        self.rng = np.random.default_rng(seed)

    @property
    def device(self) -> str:
        """Get current device."""
        return self._device

    def to(self, device: str):
        """Move PLM to specified device."""
        self._device = device
        return self

    # ============================================================================
    # EXTRACTED FROM: parseq_official_adapter.py:92-140
    # DO NOT MODIFY - Numerical equivalence required
    # ============================================================================

    def gen_tgt_perms(self, tgt):
        """
        Generate shared permutations for the whole batch.
        From official PARSeq implementation.
        """
        # We don't permute the position of BOS, we permute EOS separately
        max_num_chars = tgt.shape[1] - 2
        # Special handling for 1-character sequences
        if max_num_chars == 1:
            return torch.arange(3, device=self._device).unsqueeze(0)
        perms = [torch.arange(max_num_chars, device=self._device)] if self.perm_forward else []
        # Additional permutations if needed
        max_perms = math.factorial(max_num_chars)
        if self.perm_mirrored:
            max_perms //= 2
        num_gen_perms = min(self.max_gen_perms, max_perms)
        # For 4-char sequences and shorter, generate all permutations
        if max_num_chars < 5:
            if max_num_chars == 4 and self.perm_mirrored:
                selector = [0, 3, 4, 6, 9, 10, 12, 16, 17, 18, 19, 21]
            else:
                selector = list(range(max_perms))
            perm_pool = torch.as_tensor(
                list(permutations(range(max_num_chars), max_num_chars)),
                device=self._device,
            )[selector]
            if self.perm_forward:
                perm_pool = perm_pool[1:]
            perms = torch.stack(perms)
            if len(perm_pool):
                i = self.rng.choice(len(perm_pool), size=num_gen_perms - len(perms), replace=False)
                perms = torch.cat([perms, perm_pool[i]])
        else:
            perms.extend(
                [torch.randperm(max_num_chars, device=self._device) for _ in range(num_gen_perms - len(perms))]
            )
            perms = torch.stack(perms)
        if self.perm_mirrored:
            # Add complementary pairs
            comp = perms.flip(-1)
            perms = torch.stack([perms, comp]).transpose(0, 1).reshape(-1, max_num_chars)
        # Add position indices of BOS and EOS
        bos_idx = perms.new_zeros((len(perms), 1))
        eos_idx = perms.new_full((len(perms), 1), max_num_chars + 1)
        perms = torch.cat([bos_idx, perms + 1, eos_idx], dim=1)
        # Special handling for reverse direction
        if len(perms) > 1:
            perms[1, 1:] = max_num_chars + 1 - torch.arange(max_num_chars + 1, device=self._device)
        return perms

    # ============================================================================
    # EXTRACTED FROM: parseq_official_adapter.py:142-156
    # DO NOT MODIFY - Numerical equivalence required
    # ============================================================================

    def generate_attn_masks(self, perm):
        """
        Generate attention masks given a sequence permutation.
        From official PARSeq implementation.
        """
        sz = perm.shape[0]
        mask = torch.zeros((sz, sz), dtype=torch.bool, device=self._device)
        for i in range(sz):
            query_idx = perm[i]
            masked_keys = perm[i + 1:]
            mask[query_idx, masked_keys] = True
        content_mask = mask[:-1, :-1].clone()
        mask[torch.eye(sz, dtype=torch.bool, device=self._device)] = True  # mask "self"
        query_mask = mask[1:, :-1]
        return AttentionMasks(content_mask=content_mask, query_mask=query_mask)
