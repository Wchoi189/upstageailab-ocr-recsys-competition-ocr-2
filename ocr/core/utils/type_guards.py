"""
Runtime Type Guards for OCR Components

This module provides runtime type checking utilities that go beyond static
type hints. These guards validate tensor shapes, dtypes, and semantic
constraints at runtime.

Usage:
    >>> from ocr.core.utils.type_guards import is_valid_permutation
    >>> perm = torch.tensor([0, 2, 1, 3])
    >>> if is_valid_permutation(perm):
    ...     # Safe to use as permutation
"""

from __future__ import annotations

from typing import TypeGuard

import torch
from torch import Tensor


def is_valid_permutation(perm: Tensor) -> TypeGuard[Tensor]:
    """Validate permutation tensor structure.

    A valid permutation must:
    1. Be 1D tensor
    2. Have integer dtype (long, int32, int64)
    3. Contain values 0 to N-1 exactly once (no duplicates)

    Args:
        perm: Tensor to validate

    Returns:
        True if valid permutation, False otherwise

    Example:
        >>> perm = torch.tensor([0, 2, 1, 3])
        >>> is_valid_permutation(perm)
        True
        >>>
        >>> perm = torch.tensor([0, 1, 1, 3])  # Duplicate 1
        >>> is_valid_permutation(perm)
        False
    """
    # Check dimensionality
    if perm.ndim != 1:
        return False

    # Check dtype
    if perm.dtype not in (torch.long, torch.int32, torch.int64):
        return False

    # Check for valid permutation (0 to N-1, no duplicates)
    N = len(perm)
    sorted_perm = perm.sort()[0]
    expected = torch.arange(N, device=perm.device, dtype=perm.dtype)

    return torch.equal(sorted_perm, expected)


def is_attention_mask(mask: Tensor) -> TypeGuard[Tensor]:
    """Validate attention mask structure.

    A valid attention mask must:
    1. Be 2D tensor (square matrix)
    2. Have boolean dtype
    3. Have matching dimensions (L x L)

    Args:
        mask: Tensor to validate

    Returns:
        True if valid attention mask, False otherwise

    Example:
        >>> mask = torch.ones(5, 5, dtype=torch.bool)
        >>> is_attention_mask(mask)
        True
        >>>
        >>> mask = torch.ones(5, 6, dtype=torch.bool)  # Not square
        >>> is_attention_mask(mask)
        False
    """
    # Check dimensionality
    if mask.ndim != 2:
        return False

    # Check dtype
    if mask.dtype != torch.bool:
        return False

    # Check square
    if mask.shape[0] != mask.shape[1]:
        return False

    return True


def is_valid_logits(logits: Tensor, vocab_size: int | None = None) -> TypeGuard[Tensor]:
    """Validate logits tensor structure.

    Valid logits must:
    1. Be 3D tensor [B, L, V]
    2. Have float dtype
    3. Have non-zero dimensions
    4. Match expected vocab_size if provided

    Args:
        logits: Tensor to validate
        vocab_size: Expected vocabulary size (optional)

    Returns:
        True if valid logits, False otherwise

    Example:
        >>> logits = torch.randn(32, 25, 100)
        >>> is_valid_logits(logits, vocab_size=100)
        True
    """
    # Check dimensionality
    if logits.ndim != 3:
        return False

    # Check dtype (must be float)
    if not logits.dtype.is_floating_point:
        return False

    # Check non-zero dimensions
    B, L, V = logits.shape
    if B == 0 or L == 0 or V == 0:
        return False

    # Check vocab size if provided
    if vocab_size is not None and V != vocab_size:
        return False

    return True


def is_valid_token_sequence(tokens: Tensor, vocab_size: int | None = None) -> TypeGuard[Tensor]:
    """Validate token sequence structure.

    Valid token sequence must:
    1. Be 1D or 2D tensor
    2. Have integer dtype
    3. Have non-negative values
    4. Have values < vocab_size if provided

    Args:
        tokens: Tensor to validate
        vocab_size: Expected vocabulary size (optional)

    Returns:
        True if valid token sequence, False otherwise

    Example:
        >>> tokens = torch.tensor([[1, 5, 10, 2]])
        >>> is_valid_token_sequence(tokens, vocab_size=100)
        True
    """
    # Check dimensionality (1D or 2D)
    if tokens.ndim not in (1, 2):
        return False

    # Check dtype (must be integer)
    if tokens.dtype not in (torch.long, torch.int32, torch.int64):
        return False

    # Check non-negative
    if tokens.min() < 0:
        return False

    # Check vocab range if provided
    if vocab_size is not None and tokens.max() >= vocab_size:
        return False

    return True


def is_valid_feature_list(features: list) -> TypeGuard[list[Tensor]]:
    """Validate encoder feature list structure.

    Valid feature list must:
    1. Be non-empty list
    2. Contain only tensors
    3. All tensors have 4D shape [B, C, H, W]
    4. All tensors have same batch size

    Args:
        features: List to validate

    Returns:
        True if valid feature list, False otherwise

    Example:
        >>> features = [
        ...     torch.randn(32, 64, 64, 64),
        ...     torch.randn(32, 128, 32, 32)
        ... ]
        >>> is_valid_feature_list(features)
        True
    """
    # Check non-empty
    if not features:
        return False

    # Check all are tensors
    if not all(isinstance(f, Tensor) for f in features):
        return False

    # Check all are 4D
    if not all(f.ndim == 4 for f in features):
        return False

    # Check consistent batch size
    batch_sizes = [f.shape[0] for f in features]
    if len(set(batch_sizes)) != 1:
        return False

    return True


def validate_decoder_input(
    features: list[Tensor] | Tensor,
    targets: Tensor | None,
    mode: str
) -> tuple[bool, str]:
    """Validate decoder forward inputs comprehensively.

    This is a convenience function that validates all inputs to a decoder
    forward pass and returns both status and error message.

    Args:
        features: Encoder features (list or tensor)
        targets: Target tokens (optional)
        mode: Decoder mode string

    Returns:
        (is_valid, error_message) where:
            - is_valid: True if all inputs valid
            - error_message: Empty string if valid, error description if invalid

    Example:
        >>> features = [torch.randn(32, 256, 16, 16)]
        >>> targets = torch.randint(0, 100, (32, 25))
        >>> valid, error = validate_decoder_input(features, targets, "train")
        >>> if not valid:
        ...     raise ValueError(error)
    """
    # Validate mode
    valid_modes = {"train", "inference", "validation"}
    if mode not in valid_modes:
        return False, f"Invalid mode: {mode}, must be one of {valid_modes}"

    # Validate features
    if isinstance(features, list):
        if not is_valid_feature_list(features):
            return False, "Invalid feature list: must be non-empty list of 4D tensors with same batch size"
    elif isinstance(features, Tensor):
        if features.ndim != 3:
            return False, f"Invalid features tensor: must be 3D [B, S, D], got {features.ndim}D"
    else:
        return False, f"Invalid features type: {type(features)}, must be list[Tensor] or Tensor"

    # Validate targets based on mode
    if mode in {"train", "validation"}:
        if targets is None:
            return False, f"Mode '{mode}' requires targets, got None"
        if not is_valid_token_sequence(targets):
            return False, "Invalid targets: must be 1D or 2D integer tensor with non-negative values"

    return True, ""


# Example usage in assertions
def assert_valid_permutation(perm: Tensor, name: str = "permutation"):
    """Assert permutation is valid, raise with descriptive error."""
    if not is_valid_permutation(perm):
        raise ValueError(
            f"Invalid {name}: must be 1D integer tensor containing values 0 to N-1 exactly once. "
            f"Got shape={perm.shape}, dtype={perm.dtype}, values={perm.tolist()}"
        )


def assert_attention_mask(mask: Tensor, name: str = "attention_mask"):
    """Assert attention mask is valid, raise with descriptive error."""
    if not is_attention_mask(mask):
        raise ValueError(
            f"Invalid {name}: must be 2D square boolean tensor. "
            f"Got shape={mask.shape}, dtype={mask.dtype}"
        )


def assert_valid_logits(logits: Tensor, vocab_size: int | None = None, name: str = "logits"):
    """Assert logits are valid, raise with descriptive error."""
    if not is_valid_logits(logits, vocab_size):
        raise ValueError(
            f"Invalid {name}: must be 3D [B, L, V] float tensor. "
            f"Got shape={logits.shape}, dtype={logits.dtype}"
            + (f", expected vocab_size={vocab_size}" if vocab_size else "")
        )
