"""Timm-based backbone encoder for OCR models."""

from collections.abc import Sequence

import timm
import torch

from ..core import BaseEncoder


class TimmBackbone(BaseEncoder):
    """Timm-based backbone encoder that extracts multi-scale features.

    Uses pretrained models from the timm library to extract features at different
    scales for use in OCR detection models. Accepts a variety of configuration
    keys produced by the Streamlit UI and gracefully ignores encoder-specific
    overrides that are not relevant to the timm factory.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        select_features: Sequence[int] | None = None,
        pretrained: bool = True,
        output_indices: Sequence[int] | None = None,
        freeze_backbone: bool = False,
        **timm_kwargs,
    ):
        """Initialize the Timm backbone.

        Args:
            model_name: Name of the timm model to use (e.g., 'resnet18', 'efficientnet_b0')
            select_features: Indices of feature levels to extract. If ``None``, the
                indices are inferred from ``output_indices`` or the trailing stages
                of the backbone.
            pretrained: Whether to use pretrained weights from timm.
            output_indices: Optional alias provided by some configs for
                ``select_features``.
            freeze_backbone: If ``True``, disables gradient updates for the core
                timm backbone.
            **timm_kwargs: Additional arguments forwarded to ``timm.create_model``.
        """
        super().__init__()

        # Harmonise feature-selection aliases coming from different presets.
        if output_indices is not None:
            select_features = list(output_indices)
        elif select_features is not None:
            select_features = list(select_features)

        # Remove overrides that belong to downstream components but may be
        # injected via architecture presets (e.g., CRAFT defaults).
        timm_kwargs.pop("extra_channels", None)

        # ``timm`` expects ``features_only`` models when we want intermediate
        # activations. Enforce this regardless of UI input to avoid silent
        # misconfiguration.
        timm_kwargs.pop("features_only", None)

        out_indices_override = timm_kwargs.pop("out_indices", None)
        if out_indices_override is not None:
            select_features = list(out_indices_override)

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            **timm_kwargs,
        )

        feature_info = self.model.feature_info
        available_indices = list(range(len(feature_info)))

        if select_features is None:
            # Default to the deepest four stages when available; otherwise use
            # all produced feature maps.
            if len(available_indices) >= 4:
                select_features = available_indices[-4:]
            else:
                select_features = available_indices

        for idx in select_features:
            if idx not in available_indices:
                raise ValueError(f"Requested feature index {idx} not available. Valid indices: {available_indices}.")

        self.select_features = list(select_features)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Cache feature information for property access
        self._feature_info = feature_info
        self._out_channels = [self._feature_info[i]["num_chs"] for i in self.select_features]
        self._strides = [self._feature_info[i]["reduction"] for i in self.select_features]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract features from input images.

        BUG-20251109-002: Added validation to detect NaN values in encoder output.
        NaN values in encoder output cause CUDA illegal instruction errors downstream.
        See: docs/bug_reports/BUG-20251109-002-code-changes.md

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature tensors at selected levels
        """
        # BUG-20251109-002: Validate input before encoder forward pass
        if torch.isnan(x).any():
            raise ValueError(f"NaN values detected in encoder input. Shape: {x.shape}, Device: {x.device}")
        if torch.isinf(x).any():
            raise ValueError(f"Inf values detected in encoder input. Shape: {x.shape}, Device: {x.device}")

        features = self.model(x)

        # BUG-20251110-001: Validate encoder output to catch NaN values early
        # Move to CPU before checking to avoid CUDA illegal instruction errors
        # CUDA memory corruption can cause errors when checking NaN on GPU tensors
        try:
            if isinstance(features, (list, tuple)):
                for i, feat in enumerate(features):
                    # BUG-20251110-001: Move to CPU before checking for NaN/Inf
                    # This prevents CUDA illegal instruction errors from corrupted memory
                    try:
                        feat_cpu = feat.detach().cpu()
                        if torch.isnan(feat_cpu).any():
                            raise ValueError(
                                f"NaN values detected in TimmBackbone encoder output[{i}]. "
                                f"Shape: {feat.shape}, Device: {feat.device}. "
                                f"This suggests gradient explosion or numerical instability in the encoder."
                            )
                        if torch.isinf(feat_cpu).any():
                            raise ValueError(
                                f"Inf values detected in TimmBackbone encoder output[{i}]. "
                                f"Shape: {feat.shape}, Device: {feat.device}. "
                                f"This suggests gradient explosion or numerical instability in the encoder."
                            )
                    except RuntimeError as e:
                        # If CUDA error occurs during validation, the memory is corrupted
                        error_msg = str(e)
                        if "CUDA" in error_msg or "cuda" in error_msg or "illegal" in error_msg.lower():
                            raise RuntimeError(
                                f"CUDA error detected while validating encoder output[{i}]. "
                                f"This indicates CUDA memory corruption. "
                                f"Shape: {feat.shape}, Device: {feat.device}. "
                                f"Original error: {e}"
                            ) from e
                        raise
            else:
                # BUG-20251110-001: Move to CPU before checking for NaN/Inf
                try:
                    features_cpu = features.detach().cpu()
                    if torch.isnan(features_cpu).any():
                        raise ValueError(
                            f"NaN values detected in TimmBackbone encoder output. "
                            f"Shape: {features.shape}, Device: {features.device}. "
                            f"This suggests gradient explosion or numerical instability in the encoder."
                        )
                    if torch.isinf(features_cpu).any():
                        raise ValueError(
                            f"Inf values detected in TimmBackbone encoder output. "
                            f"Shape: {features.shape}, Device: {features.device}. "
                            f"This suggests gradient explosion or numerical instability in the encoder."
                        )
                except RuntimeError as e:
                    # If CUDA error occurs during validation, the memory is corrupted
                    error_msg = str(e)
                    if "CUDA" in error_msg or "cuda" in error_msg or "illegal" in error_msg.lower():
                        raise RuntimeError(
                            f"CUDA error detected while validating encoder output. "
                            f"This indicates CUDA memory corruption. "
                            f"Shape: {features.shape}, Device: {features.device}. "
                            f"Original error: {e}"
                        ) from e
                    raise
        except RuntimeError as e:
            # Catch any CUDA errors that occur during validation
            error_msg = str(e)
            if "CUDA" in error_msg or "cuda" in error_msg or "illegal" in error_msg.lower():
                raise RuntimeError(
                    f"CUDA error detected in TimmBackbone encoder validation. "
                    f"This indicates CUDA memory corruption. "
                    f"Original error: {e}"
                ) from e
            raise

        return [features[i] for i in self.select_features]

    @property
    def out_channels(self) -> list[int]:
        """Return the number of output channels for each feature level."""
        return self._out_channels

    @property
    def strides(self) -> list[int]:
        """Return the stride (downsampling factor) for each feature level."""
        return self._strides
