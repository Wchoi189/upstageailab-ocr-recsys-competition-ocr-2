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

        try:
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                **timm_kwargs,
            )
            self._is_features_only = True
        except RuntimeError as e:
            if "features_only not implemented" in str(e):
                self.model = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    features_only=False,
                    **timm_kwargs,
                )
                self._is_features_only = False
            else:
                raise e

        if self._is_features_only:
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
            self._feature_info = feature_info
            self._out_channels = [self._feature_info[i]["num_chs"] for i in self.select_features]
            self._strides = [self._feature_info[i]["reduction"] for i in self.select_features]
        else:
            # Fallback for models like ViT that don't support features_only
            # We assume a single output scale (the sequence of embeddings)
            self.select_features = [-1] # Dummy index

            # Try to get embed_dim or num_features
            out_ch = getattr(self.model, "embed_dim", None) or getattr(self.model, "num_features", None)
            if out_ch is None:
                raise RuntimeError(f"Could not determine output channels for model {model_name}")

            self._out_channels = [out_ch]

            # Try to determine stride usually it's patch size e.g. 16
            stride = 16 # Default fallback
            if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "patch_size"):
                 patch_size = self.model.patch_embed.patch_size
                 if isinstance(patch_size, tuple):
                     stride = patch_size[0]
                 else:
                     stride = patch_size
            self._strides = [stride]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features from input images.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            List of feature tensors at selected levels
        """
        if getattr(self, "_is_features_only", True):
             features = self.model(x)
             return [features[i] for i in self.select_features]
        else:
             # ViT case: forward_features returns the sequence of patch embeddings (B, N, E)
             features = self.model.forward_features(x)
             return [features]

    @property
    def out_channels(self) -> list[int]:
        """Return the number of output channels for each feature level."""
        return self._out_channels

    @property
    def strides(self) -> list[int]:
        """Return the stride (downsampling factor) for each feature level."""
        return self._strides
