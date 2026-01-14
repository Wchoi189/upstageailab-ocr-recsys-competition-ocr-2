from typing import Any
import logging

import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate

from ocr.core.utils.config_utils import ensure_dict, is_config

from ocr.core import get_registry

logger = logging.getLogger(__name__)


class OCRModel(nn.Module):
    # Keys that actually affect the loss computation (avoids torch.compile overhead)
    LOSS_KEYS: set[str] = {"prob_mask", "thresh_mask", "prob_maps", "thresh_maps"}

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Hydra moved from "architectures" -> "architecture_name"; honour both for backwards compatibility.
        self.architecture_name = getattr(cfg, "architecture_name", None) or getattr(cfg, "architectures", None)

        self.architecture_config_obj = None
        if is_config(self.architecture_name):
             # Handle case where architecture_name is a Config object (DictConfig or dict)
             # This happens when using Hydra composition like 'model/architectures: parseq'
             self.architecture_config_obj = self.architecture_name
             self.architecture_name = self.architecture_config_obj.get("architecture_name") or self.architecture_config_obj.get("name")

        if self.architecture_name:
            self._init_from_registry(cfg)
        else:
            self._init_from_components(cfg)

    def forward(self, images, return_loss=True, **kwargs):
        encoded_features = self.encoder(images)
        decoded_features = self.decoder(encoded_features)
        pred = self.head(decoded_features, return_loss)

        # Loss 계산
        if return_loss:
            # Extract ground truth from kwargs
            gt_binary = kwargs.get("prob_maps")
            gt_thresh = kwargs.get("thresh_maps")

            # Static filtering of kwargs to prevent torch.compile re-tracing
            # Optimization: Direct dict comprehension is faster than multiple if checks inside loop
            loss_input_kwargs = {k: v for k, v in kwargs.items() if k in self.LOSS_KEYS}

            if gt_binary is not None and gt_thresh is not None:
                loss, loss_dict = self.loss(pred, gt_binary, gt_thresh, **loss_input_kwargs)
            else:
                loss, loss_dict = self.loss(pred, **loss_input_kwargs)

            pred.update(loss=loss, loss_dict=loss_dict)

        return pred

    def get_optimizers(self):
        optimizer_config = self.cfg.optimizer
        optimizer = instantiate(optimizer_config, params=self.parameters())

        scheduler = None
        if "scheduler" in self.cfg:
            scheduler_config = self.cfg.scheduler
            scheduler = instantiate(scheduler_config, optimizer=optimizer)

        return [optimizer], [scheduler] if scheduler else []

    def get_polygons_from_maps(
        self,
        batch: dict[str, Any],
        pred: dict[str, torch.Tensor]
    ) -> tuple[list[list[list[int]]], list[list[float]]]:
        """Delegate to head's polygon extraction."""
        return self.head.get_polygons_from_maps(batch, pred)

    def _init_from_registry(self, cfg):
        registry = get_registry()
        component_configs = self._prepare_component_configs(cfg)
        architecture_name = str(self.architecture_name)

        # Logic: registry is the source of truth for component creation
        components = registry.create_architecture_components(architecture_name, **component_configs)

        self.encoder = components["encoder"]
        self.decoder = components["decoder"]
        self.head = components["head"]
        self.loss = components["loss"]

    def _init_from_components(self, cfg):
        from ocr.core.models.decoder import get_decoder_by_cfg
        from ocr.core.models.encoder import get_encoder_by_cfg
        from ocr.core.models.head import get_head_by_cfg
        from ocr.core.models.loss import get_loss_by_cfg

        self.encoder = get_encoder_by_cfg(cfg.encoder)

        decoder_cfg = cfg.decoder.copy()
        if hasattr(decoder_cfg, "in_channels") or "in_channels" in decoder_cfg:
            encoder_out_channels = self.encoder.out_channels
            decoder_cfg.in_channels = encoder_out_channels

        self.decoder = get_decoder_by_cfg(decoder_cfg)
        self.head = get_head_by_cfg(cfg.head)
        self.loss = get_loss_by_cfg(cfg.loss)

    def _prepare_component_configs(self, cfg) -> dict[str, Any]:
        """
        Prepare component configurations with correct precedence using OmegaConf.

        Hierarchy (lowest to highest):
        1. Architecture Defaults (from registry/hydra composition)
        2. Legacy/Experiment Top-level (cfg.encoder, etc.) - FILTERED for conflicts
        3. User Explicit Overrides (cfg.component_overrides) - FILTERED for conflicts
        """
        # Start with an empty structured config
        base = OmegaConf.create({})

        # 1. Arch Defaults (if using Hydra composition)
        if self.architecture_config_obj:
            # Merge arch-specific overrides first
            arch_overrides = getattr(self.architecture_config_obj, "component_overrides", {})
            if arch_overrides:
                 base = OmegaConf.merge(base, arch_overrides)

            # Merge direct definitions (encoder: {name: ...})
            for k in ["encoder", "decoder", "head", "loss"]:
                val = getattr(self.architecture_config_obj, k, None)
                if val is None and is_config(self.architecture_config_obj):
                    val = self.architecture_config_obj.get(k)

                if val is not None:
                     # If the key exists in base, merge it. If not, set it.
                     # We use merge to support partial updates if base already has content
                     current = base.get(k, OmegaConf.create({}))
                     base[k] = OmegaConf.merge(current, val)

        # 2. Legacy/Experiment Overrides (e.g., cfg.encoder in your main yaml)
        # These are commonly found in _base/model/ definitions.
        legacy_cfg = OmegaConf.create({})
        for k in ["encoder", "decoder", "head", "loss"]:
             val = getattr(cfg, k, None)
             if val is not None:
                 legacy_cfg[k] = val

        # BUG_003: We filter these if they change the 'name' of the component
        # defined in the architecture, unless the user explicitly wants that.
        filtered_legacy = self._filter_conflicts(legacy_cfg, base)
        base = OmegaConf.merge(base, filtered_legacy)

        # 3. User Explicit Overrides (highest priority)
        user_overrides = getattr(cfg, "component_overrides", {})
        if user_overrides:
             user_overrides_conf = ensure_dict(user_overrides) # Ensure it's dict-like for key access
             filtered_user_overrides = self._filter_conflicts(user_overrides_conf, base)
             base = OmegaConf.merge(base, filtered_user_overrides)

        return self._format_for_registry(base)

    def _filter_conflicts(self, incoming: dict, existing: DictConfig) -> dict:
        """Removes incoming configs if they attempt to change a component's
        identity (name) without being an explicit override.
        """
        filtered = {}
        incoming_dict = ensure_dict(incoming)

        for k, v in incoming_dict.items():
            if k not in ["encoder", "decoder", "head", "loss"]:
                filtered[k] = v
                continue

            # Check for name conflict
            existing_comp = existing.get(k)
            # Existing name might be in {name: ...} or just implied if missing
            existing_name = existing_comp.get("name") if existing_comp and is_config(existing_comp) else None

            incoming_comp = v
            incoming_name = incoming_comp.get("name") if incoming_comp and is_config(incoming_comp) else None

            if existing_name and incoming_name and existing_name != incoming_name:
                logger.info(
                    f"BUG_003: Filtering legacy/override {k} ({incoming_name}) "
                    f"in favor of architecture {k} ({existing_name})"
                )
                continue

            filtered[k] = v

        return filtered

    def _format_for_registry(self, merged_cfg: DictConfig) -> dict[str, Any]:
        """Converts internal DictConfig to the flat dict format the registry expects."""
        out = {}
        overrides = ensure_dict(merged_cfg)

        for name in ("encoder", "decoder", "head", "loss"):
            section = overrides.get(name)
            if section is None:
                continue

            section = ensure_dict(section)

            # Support both {name: '...', params: {...}} and flat {param1: val} formats
            if "name" in section:
                out[f"{name}_name"] = section["name"]

                # Get Config/Params
                params = section.get("params")
                if params is None:
                    # If no 'params' key, assume everything else is params
                    params = {k: v for k, v in section.items() if k != "name"}

                out[f"{name}_config"] = ensure_dict(params)
            else:
                # Flat config, assume it's just params for the default component
                out[f"{name}_config"] = section

        return out
