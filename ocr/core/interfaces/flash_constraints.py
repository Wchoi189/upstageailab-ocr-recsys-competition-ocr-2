"""
Flash Attention Constraints and Configuration

This module defines type-safe configuration and validation for Flash Attention
integration. Flash Attention has strict hardware and precision requirements
that must be validated at runtime.

References:
    - Flash Attention paper: https://arxiv.org/abs/2205.14135
    - PyTorch docs: torch.nn.functional.scaled_dot_product_attention
    - Hardware requirements: Ampere+ (sm_80+) for optimal performance
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class FlashAttentionConfig:
    """Configuration and constraints for Flash Attention.

    Flash Attention provides 2-4x speedup over standard attention but has
    strict requirements:
    1. Data type: fp16 or bfloat16 only (no fp32)
    2. Head dimension: Must be multiple of 8
    3. PyTorch version: ≥2.0 (≥2.2 for FlashAttention-2)
    4. Hardware: Ampere+ GPUs (sm_80+) for auto-selection

    Attributes:
        d_model: Model dimension (hidden size)
        nhead: Number of attention heads
        enabled: Whether Flash Attention is enabled
        force_fp16: Force fp16 precision (for compatibility)
        fallback_on_error: Fall back to standard attention if Flash fails

    Validation:
        All constraints are checked in __post_init__ and raise ValueError
        if violated. Use is_compatible to check without raising.

    Example:
        >>> config = FlashAttentionConfig(d_model=384, nhead=12)
        >>> if config.is_compatible:
        ...     # Use Flash Attention
        >>> else:
        ...     # Fall back to standard attention
    """
    d_model: int
    nhead: int
    enabled: bool = True
    force_fp16: bool = True
    fallback_on_error: bool = True

    def __post_init__(self):
        # Basic validation
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.nhead <= 0:
            raise ValueError(f"nhead must be positive, got {self.nhead}")

        # Validate d_model divisibility
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead ({self.nhead})"
            )

        # If enabled, validate Flash Attention constraints
        if self.enabled:
            self._validate_flash_constraints()

    def _validate_flash_constraints(self):
        """Validate Flash Attention specific constraints."""
        head_dim = self.d_model // self.nhead

        # Constraint 1: Head dimension must be multiple of 8
        if head_dim % 8 != 0:
            raise ValueError(
                f"Flash Attention requires head_dim to be multiple of 8. "
                f"Got d_model={self.d_model}, nhead={self.nhead}, "
                f"head_dim={head_dim}. "
                f"Try adjusting d_model or nhead. "
                f"Examples: d_model=384 with nhead=12 (head_dim=32 ✓), "
                f"d_model=512 with nhead=8 (head_dim=64 ✓)"
            )

        # Constraint 2: PyTorch version
        torch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
        if torch_version < (2, 0):
            raise RuntimeError(
                f"Flash Attention requires PyTorch ≥2.0, got {torch.__version__}. "
                f"Please upgrade: pip install torch>=2.0"
            )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.d_model // self.nhead

    @property
    def is_compatible(self) -> bool:
        """Check if Flash Attention is compatible with current setup.

        Returns:
            True if Flash Attention can be used, False otherwise.

        Compatibility requires:
            - enabled=True
            - CUDA available
            - Ampere+ GPU (sm_80+)
            - Valid head_dim (multiple of 8)
            - PyTorch ≥2.0
        """
        if not self.enabled:
            return False

        # Check CUDA availability
        if not torch.cuda.is_available():
            return False

        # Check GPU compute capability (Ampere = sm_80+)
        try:
            capability = torch.cuda.get_device_capability()
            major, minor = capability
            compute_capability = major * 10 + minor

            # Ampere: sm_80, sm_86
            # Ada Lovelace: sm_89
            # Hopper: sm_90
            if compute_capability < 80:
                return False
        except RuntimeError:
            return False

        # Check constraints without raising
        try:
            self._validate_flash_constraints()
            return True
        except (ValueError, RuntimeError):
            return False

    @property
    def gpu_architecture(self) -> str | None:
        """Get GPU architecture name if CUDA available."""
        if not torch.cuda.is_available():
            return None

        try:
            capability = torch.cuda.get_device_capability()
            major, minor = capability

            # Map compute capability to architecture
            arch_map = {
                75: "Turing (RTX 20xx)",
                80: "Ampere (A100, RTX 30xx)",
                86: "Ampere (A6000, RTX A-series)",
                89: "Ada Lovelace (RTX 40xx)",
                90: "Hopper (H100)",
            }

            compute_capability = major * 10 + minor
            return arch_map.get(compute_capability, f"Unknown (sm_{compute_capability})")
        except RuntimeError:
            return None

    def get_recommended_dtype(self) -> torch.dtype:
        """Get recommended dtype for Flash Attention.

        Returns:
            torch.float16 if force_fp16=True or Ampere GPU
            torch.bfloat16 if Ada Lovelace+ GPU and not forced fp16
        """
        if self.force_fp16:
            return torch.float16

        # bfloat16 is better on newer GPUs
        if torch.cuda.is_available():
            try:
                capability = torch.cuda.get_device_capability()
                major = capability[0]
                # Ada Lovelace+ supports bfloat16 better
                if major >= 8:  # Ampere and newer
                    return torch.bfloat16
            except RuntimeError:
                pass

        return torch.float16

    def __str__(self) -> str:
        """Human-readable configuration summary."""
        status = "✓ Compatible" if self.is_compatible else "✗ Incompatible"
        return (
            f"FlashAttentionConfig(\n"
            f"  d_model={self.d_model}, nhead={self.nhead}, head_dim={self.head_dim}\n"
            f"  enabled={self.enabled}, {status}\n"
            f"  GPU: {self.gpu_architecture or 'N/A'}\n"
            f"  Recommended dtype: {self.get_recommended_dtype()}\n"
            f")"
        )


def create_flash_config(
    d_model: int,
    nhead: int,
    enabled: bool = True,
    auto_detect: bool = True
) -> FlashAttentionConfig:
    """Create Flash Attention config with automatic compatibility detection.

    This is the recommended way to create a Flash Attention config as it
    automatically disables Flash if the system is incompatible.

    Args:
        d_model: Model dimension
        nhead: Number of attention heads
        enabled: User preference to enable Flash (can be overridden by auto_detect)
        auto_detect: Automatically disable if system incompatible

    Returns:
        FlashAttentionConfig with enabled status based on compatibility

    Example:
        >>> # Automatic detection
        >>> config = create_flash_config(d_model=384, nhead=12)
        >>> print(config)  # Shows compatibility status
        >>>
        >>> # Force disable
        >>> config = create_flash_config(d_model=384, nhead=12, enabled=False)
        >>> assert not config.enabled
    """
    config = FlashAttentionConfig(
        d_model=d_model,
        nhead=nhead,
        enabled=enabled
    )

    # Auto-disable if incompatible
    if auto_detect and enabled and not config.is_compatible:
        import warnings
        warnings.warn(
            f"Flash Attention is incompatible with current setup. "
            f"Falling back to standard attention. "
            f"Config: {config}",
            UserWarning
        )
        # Recreate with disabled
        config = FlashAttentionConfig(
            d_model=d_model,
            nhead=nhead,
            enabled=False
        )

    return config


# Type aliases
FlashAttentionDtype = Literal[torch.float16, torch.bfloat16]
