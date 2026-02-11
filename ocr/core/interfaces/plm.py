"""
Permutation Language Modeling (PLM) Data Contracts

This module defines type-safe interfaces and data structures for PLM components.
PLM is the core training strategy in PARSeq that uses K random permutations
of token orderings instead of left-to-right only.

References:
    - PARSeq paper: https://arxiv.org/abs/2207.06966
    - Implementation: ocr/domains/recognition/models/parseq_official_adapter.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


@dataclass(frozen=True)
class PLMConfig:
    """Configuration for Permutation Language Modeling.

    Attributes:
        max_len: Maximum sequence length (excluding BOS/EOS)
        perm_num: Number of permutations to generate (K in paper)
        perm_forward: Whether to include forward (left-to-right) permutation
        perm_mirrored: Whether to generate mirrored (complementary) pairs
            If True, generates K/2 base + K/2 reversed permutations

    Constraints:
        - max_len > 0
        - perm_num > 0
        - If perm_mirrored=True, perm_num must be even

    Example:
        >>> config = PLMConfig(max_len=25, perm_num=6, perm_mirrored=True)
        >>> # Generates 3 base permutations + 3 mirrored permutations
    """
    max_len: int = 25
    perm_num: int = 6
    perm_forward: bool = True
    perm_mirrored: bool = True

    def __post_init__(self):
        if self.max_len <= 0:
            raise ValueError(f"max_len must be positive, got {self.max_len}")
        if self.perm_num <= 0:
            raise ValueError(f"perm_num must be positive, got {self.perm_num}")
        if self.perm_mirrored and self.perm_num % 2 != 0:
            raise ValueError(
                f"perm_num must be even when perm_mirrored=True, got {self.perm_num}"
            )

    @property
    def max_gen_perms(self) -> int:
        """Number of base permutations to generate before mirroring."""
        return self.perm_num // 2 if self.perm_mirrored else self.perm_num


@dataclass(frozen=True)
class AttentionMasks:
    """Attention masks for permutation-based decoding.

    In PLM, attention masks enforce causal structure based on the permutation
    ordering. Two masks are needed:
    - content_mask: For context tokens (everything except last token)
    - query_mask: For query tokens (everything except first token)

    Attributes:
        content_mask: Boolean mask [L-1, L-1] for context attention
            True = masked (no attention), False = attend
        query_mask: Boolean mask [L-1, L-1] for query attention
            True = masked (no attention), False = attend

    Shape Convention:
        - L is sequence length (including BOS/EOS)
        - Masks are [L-1, L-1] because we process L-1 positions
        - PyTorch convention: True = ignore, False = attend

    Example:
        >>> # For sequence of length 5: [BOS, a, b, c, EOS]
        >>> # content_mask covers positions [BOS, a, b, c]
        >>> # query_mask covers positions [a, b, c, EOS]
        >>> masks = AttentionMasks(
        ...     content_mask=torch.ones(4, 4, dtype=torch.bool),
        ...     query_mask=torch.ones(4, 4, dtype=torch.bool)
        ... )
    """
    content_mask: Tensor
    query_mask: Tensor

    def __post_init__(self):
        # Validate shapes match
        if self.content_mask.shape != self.query_mask.shape:
            raise ValueError(
                f"Mask shapes must match: content={self.content_mask.shape}, "
                f"query={self.query_mask.shape}"
            )

        # Validate dtype
        if self.content_mask.dtype != torch.bool:
            raise TypeError(
                f"content_mask must be bool, got {self.content_mask.dtype}"
            )
        if self.query_mask.dtype != torch.bool:
            raise TypeError(
                f"query_mask must be bool, got {self.query_mask.dtype}"
            )

        # Validate square
        if self.content_mask.ndim != 2:
            raise ValueError(
                f"Masks must be 2D, got {self.content_mask.ndim}D"
            )
        if self.content_mask.shape[0] != self.content_mask.shape[1]:
            raise ValueError(
                f"Masks must be square, got {self.content_mask.shape}"
            )

    @property
    def sequence_length(self) -> int:
        """Length of sequence these masks apply to (excluding BOS/EOS trim)."""
        return self.content_mask.shape[0]


@runtime_checkable
class PLMModule(Protocol):
    """Protocol for Permutation Language Modeling components.

    This protocol defines the interface that PLM implementations must satisfy.
    It ensures type safety and enables runtime isinstance checks.

    Usage:
        >>> from ocr.domains.recognition.models.plm import PermutationLanguageModeling
        >>> plm = PermutationLanguageModeling(max_label_length=25)
        >>> assert isinstance(plm, PLMModule)  # Runtime check

    Implementation Requirements:
        1. gen_tgt_perms: Must handle 1-char, ≤4-char, and >4-char cases
        2. generate_attn_masks: Must return valid AttentionMasks
        3. Device handling: Must support .to(device) for GPU/CPU
    """

    def gen_tgt_perms(self, tgt: Tensor) -> Tensor:
        """Generate K permutations for target sequence.

        This is the core PLM operation. Given a target sequence, it generates
        K different orderings for training with diverse decoding paths.

        Special Cases:
            - 1-char: Returns identity permutation [BOS, char, EOS]
            - ≤4-char: Uses exhaustive permutations with hardcoded selector
            - >4-char: Random sampling from all possible permutations

        Args:
            tgt: Target token indices [B, L] where L includes BOS/EOS

        Returns:
            perms: Permutation indices [K, L] where:
                - K = perm_num (number of permutations)
                - L = sequence length (including BOS/EOS)
                - perms[i] is a permutation of range(L)
                - All permutations start with BOS=0, end with EOS=max_num_chars+1

        Example:
            >>> tgt = torch.tensor([[1, 5, 10, 7, 2]])  # [BOS, a, b, c, EOS]
            >>> perms = plm.gen_tgt_perms(tgt)
            >>> perms.shape  # [6, 5] for K=6 permutations
            torch.Size([6, 5])
            >>> perms[:, 0]  # All start with BOS
            tensor([0, 0, 0, 0, 0, 0])
            >>> perms[:, -1]  # All end with EOS
            tensor([2, 2, 2, 2, 2, 2])
        """
        ...

    def generate_attn_masks(self, perm: Tensor) -> AttentionMasks:
        """Generate attention masks for a permutation.

        Given a permutation, creates causal attention masks that enforce
        the ordering specified by the permutation. This prevents the model
        from "cheating" by looking ahead in the sequence.

        Args:
            perm: Single permutation indices [L] for one sequence

        Returns:
            AttentionMasks containing:
                - content_mask: [L-1, L-1] for context tokens
                - query_mask: [L-1, L-1] for query tokens

        Mask Semantics:
            - mask[i, j] = True means position i cannot attend to position j
            - Enforces causal structure based on permutation ordering
            - Prevents look-ahead within the permutation

        Example:
            >>> perm = torch.tensor([0, 3, 1, 2, 4])  # BOS, c, a, b, EOS
            >>> masks = plm.generate_attn_masks(perm)
            >>> # Position 'a' can only attend to 'BOS' and 'c' (earlier in perm)
            >>> # Position 'b' can attend to 'BOS', 'c', and 'a'
        """
        ...


@dataclass(frozen=True)
class PLMLossConfig:
    """Configuration for PLM loss computation.

    PLM loss aggregation has specific requirements that differ from
    standard autoregressive loss:
    - Loss is weighted by character count
    - EOS tokens are removed after 2nd permutation
    - Loss is normalized by total character count across all permutations

    Attributes:
        pad_id: Padding token ID (ignored in loss)
        eos_id: End-of-sequence token ID (removed after 2nd perm)
        eos_removal_after: Which permutation to remove EOS after (default: 1 = after 2nd)

    Critical:
        The eos_removal_after=1 default means EOS is removed after the 2nd
        permutation (index 1). This is a core PARSeq training detail that
        prevents over-weighting of EOS tokens.

        DO NOT change this without understanding the implications!
        See: parseq_official_adapter.py:243-248
    """
    pad_id: int = 0
    eos_id: int = 2
    eos_removal_after: int = 1  # After 2nd permutation (0-indexed)

    def __post_init__(self):
        if self.pad_id == self.eos_id:
            raise ValueError("pad_id and eos_id must be different")
        if self.eos_removal_after < 0:
            raise ValueError("eos_removal_after must be non-negative")


# Type aliases for clarity
PermutationTensor = Tensor  # [K, L] - K permutations of length L
AttentionMaskTensor = Tensor  # [L, L] - Boolean attention mask
