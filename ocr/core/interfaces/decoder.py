"""
Decoder Interface Contracts

This module defines type-safe interfaces for autoregressive decoders that
support multiple operating modes (training, inference, validation).

The key insight is that decoders need different behaviors at train vs inference time:
- Training: Process all permutations with teacher forcing
- Inference: Autoregressive generation with beam search/greedy
- Validation: Hybrid mode (may use reduced permutations)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor


class DecoderMode(str, Enum):
    """Decoder operating modes.

    Attributes:
        TRAIN: Training with PLM permutations and teacher forcing
            - Uses K permutations per sample
            - Targets provided, loss computed
            - Gradients computed for backprop

        INFERENCE: Autoregressive inference/prediction
            - Single left-to-right pass
            - No targets provided
            - Greedy or beam search decoding

        VALIDATION: Validation mode (optional, same as INFERENCE usually)
            - May use reduced permutations for efficiency
            - Targets provided for metrics but not used in forward
            - Used during validation loops

    Usage:
        >>> mode = DecoderMode.TRAIN
        >>> if mode == DecoderMode.TRAIN:
        ...     # Use PLM permutations
        >>> elif mode == DecoderMode.INFERENCE:
        ...     # Use autoregressive decoding
    """
    TRAIN = "train"
    INFERENCE = "inference"
    VALIDATION = "validation"

    @property
    def requires_targets(self) -> bool:
        """Whether this mode requires target sequences."""
        return self in (DecoderMode.TRAIN, DecoderMode.VALIDATION)

    @property
    def uses_teacher_forcing(self) -> bool:
        """Whether this mode uses teacher forcing."""
        return self == DecoderMode.TRAIN


@dataclass
class DecoderOutput:
    """Standardized decoder output.

    All decoders must return this structure to ensure consistent interfaces
    across different decoder implementations.

    Attributes:
        logits: Predicted logits [B, L, vocab_size] or [B*K, L, vocab_size]
            - Training: [B*K, L, vocab_size] where K = num_permutations
            - Inference: [B, L, vocab_size]

        loss: Optional loss tensor (scalar)
            - Present in TRAIN mode
            - None in INFERENCE mode

        metadata: Optional metadata dictionary
            - Can include: attention weights, hidden states, etc.
            - Used for debugging and visualization

    Shape Conventions:
        - B: Batch size
        - L: Sequence length
        - K: Number of permutations (training only)
        - vocab_size: Vocabulary size

    Example:
        >>> # Training output
        >>> output = DecoderOutput(
        ...     logits=torch.randn(64*6, 25, 100),  # B=64, K=6, L=25, V=100
        ...     loss=torch.tensor(2.5),
        ...     metadata={"num_permutations": 6}
        ... )
        >>>
        >>> # Inference output
        >>> output = DecoderOutput(
        ...     logits=torch.randn(64, 25, 100),  # B=64, L=25, V=100
        ...     loss=None
        ... )
    """
    logits: Tensor
    loss: Tensor | None = None
    metadata: dict | None = None

    def __post_init__(self):
        # Validate logits shape
        if self.logits.ndim != 3:
            raise ValueError(
                f"logits must be 3D [B, L, V] or [B*K, L, V], got {self.logits.ndim}D"
            )

        # Validate loss shape (must be scalar if present)
        if self.loss is not None:
            if self.loss.ndim != 0:
                raise ValueError(
                    f"loss must be scalar (0D), got {self.loss.ndim}D: {self.loss.shape}"
                )

    @property
    def batch_size_with_perms(self) -> int:
        """Batch size (possibly multiplied by K permutations)."""
        return self.logits.shape[0]

    @property
    def sequence_length(self) -> int:
        """Sequence length."""
        return self.logits.shape[1]

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        return self.logits.shape[2]

    def get_predictions(self) -> Tensor:
        """Get predicted token indices [B, L] or [B*K, L]."""
        return self.logits.argmax(dim=-1)


@runtime_checkable
class AutoregressiveDecoder(Protocol):
    """Protocol for autoregressive decoders supporting multiple modes.

    This protocol defines the interface for decoders that can operate in
    both training (with PLM) and inference (autoregressive) modes.

    Key Requirements:
        1. Must support mode switching via `mode` parameter
        2. Must return standardized `DecoderOutput`
        3. Must handle both list[Tensor] and Tensor for features
        4. Must support optional targets (required for training)

    Implementation Notes:
        - TRAIN mode: Requires targets, returns loss
        - INFERENCE mode: No targets, no loss, autoregressive generation
        - Features can be either list[Tensor] (from backbone) or Tensor (pre-processed)

    Usage:
        >>> decoder = PARSeqDecoder(...)
        >>>
        >>> # Training
        >>> output = decoder(
        ...     features=encoder_output,
        ...     targets=tokens,
        ...     mode=DecoderMode.TRAIN
        ... )
        >>> loss = output.loss
        >>>
        >>> # Inference
        >>> output = decoder(
        ...     features=encoder_output,
        ...     mode=DecoderMode.INFERENCE
        ... )
        >>> predictions = output.get_predictions()
    """

    def forward(
        self,
        features: list[Tensor] | Tensor,
        targets: Tensor | None = None,
        mode: DecoderMode = DecoderMode.TRAIN,
        **kwargs
    ) -> DecoderOutput:
        """Unified forward supporting train/inference modes.

        Args:
            features: Encoder features, either:
                - list[Tensor]: Multi-scale features from backbone
                - Tensor: Pre-processed memory [B, S, D]

            targets: Target token indices [B, L] (optional)
                - Required for mode=TRAIN
                - Optional for mode=INFERENCE (ignored if provided)
                - Includes BOS and EOS tokens

            mode: Operating mode (TRAIN, INFERENCE, VALIDATION)
                - Determines decoder behavior

            **kwargs: Additional arguments, may include:
                - memory_key_padding_mask: [B, S] mask for encoder features
                - plm_config: Override PLM configuration
                - beam_size: For beam search inference

        Returns:
            DecoderOutput containing:
                - logits: Predicted logits
                - loss: Training loss (if mode=TRAIN)
                - metadata: Optional debug info

        Raises:
            ValueError: If targets not provided when required
            RuntimeError: If mode not supported

        Example:
            >>> # Training forward
            >>> output = decoder.forward(
            ...     features=encoder_features,
            ...     targets=target_tokens,
            ...     mode=DecoderMode.TRAIN
            ... )
            >>> assert output.loss is not None
            >>>
            >>> # Inference forward
            >>> output = decoder.forward(
            ...     features=encoder_features,
            ...     mode=DecoderMode.INFERENCE
            ... )
            >>> predictions = output.get_predictions()
        """
        ...


@dataclass(frozen=True)
class DecoderConfig:
    """Base configuration for autoregressive decoders.

    This provides common configuration parameters shared across different
    decoder implementations.

    Attributes:
        d_model: Model dimension (hidden size)
        nhead: Number of attention heads
        num_layers: Number of decoder layers
        dim_feedforward: Feedforward network dimension
        dropout: Dropout probability
        vocab_size: Size of token vocabulary
        max_len: Maximum sequence length
        pad_token_id: Padding token ID
        bos_token_id: Beginning-of-sequence token ID
        eos_token_id: End-of-sequence token ID

    Constraints:
        - d_model must be divisible by nhead
        - All token IDs must be distinct
        - max_len > 0
    """
    d_model: int = 384
    nhead: int = 12
    num_layers: int = 12
    dim_feedforward: int = 1536
    dropout: float = 0.1
    vocab_size: int | None = None
    max_len: int = 25
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self):
        # Validate d_model divisibility
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )

        # Validate token IDs are distinct
        token_ids = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        if len(token_ids) != 3:
            raise ValueError(
                f"Token IDs must be distinct: pad={self.pad_token_id}, "
                f"bos={self.bos_token_id}, eos={self.eos_token_id}"
            )

        # Validate positive values
        if self.max_len <= 0:
            raise ValueError(f"max_len must be positive, got {self.max_len}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")
        if self.nhead <= 0:
            raise ValueError(f"nhead must be positive, got {self.nhead}")

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.nhead
