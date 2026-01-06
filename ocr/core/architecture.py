from typing import Any, cast
import logging

import torch
from omegaconf import OmegaConf
import torch.nn as nn
from hydra.utils import instantiate

from ocr.utils.config_utils import ensure_dict, is_config

from ocr.core import get_registry
from ocr.models.decoder import get_decoder_by_cfg
from ocr.models.encoder import get_encoder_by_cfg
from ocr.models.head import get_head_by_cfg
from ocr.models.loss import get_loss_by_cfg

logger = logging.getLogger(__name__)


class OCRModel(nn.Module):
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
            if gt_binary is not None and gt_thresh is not None:
                # Filter kwargs to only pass computation-relevant parameters to avoid torch.compile recompilation
                # due to changing metadata like image_filename
                loss_kwargs = {k: v for k, v in kwargs.items() if k in {"prob_mask", "thresh_mask"}}
                loss, loss_dict = self.loss(pred, gt_binary, gt_thresh, **loss_kwargs)
            else:
                # Fallback for cases where ground truth is not available
                # Filter kwargs for loss function to avoid torch.compile issues
                loss_kwargs = {k: v for k, v in kwargs.items() if k in {"prob_mask", "thresh_mask"}}
                loss, loss_dict = self.loss(pred, **loss_kwargs)
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
        components = registry.create_architecture_components(architecture_name, **component_configs)
        self.encoder = components["encoder"]
        self.decoder = components["decoder"]
        self.head = components["head"]
        self.loss = components["loss"]

    def _init_from_components(self, cfg):
        self.encoder = get_encoder_by_cfg(cfg.encoder)

        decoder_cfg = cfg.decoder.copy()
        if hasattr(decoder_cfg, "in_channels") or "in_channels" in decoder_cfg:
            encoder_out_channels = self.encoder.out_channels
            decoder_cfg.in_channels = encoder_out_channels

        self.decoder = get_decoder_by_cfg(decoder_cfg)
        self.head = get_head_by_cfg(cfg.head)
        self.loss = get_loss_by_cfg(cfg.loss)

    def _prepare_component_configs(self, cfg) -> dict:
        """Prepare component configurations with correct precedence.

        BUG_003 FIX: Merge order and conflict filtering ensure architecture
        components take precedence over legacy defaults.

        Precedence (lowest to highest):
        - P1: Empty base
        - P2: arch_overrides (from architecture's component_overrides)
        - P3: filtered_top_level (legacy cfg.encoder/decoder/etc, filtered)
        - P4: direct_overrides (architecture's encoder/decoder/head/loss)
        - P5: cfg.component_overrides (user explicit overrides)
        """
        arch_overrides = None
        direct_overrides = {}

        # Priority 1: Check self.architecture_config_obj (from Hydra composition)
        # Use a DictConfig to accumulate merges
        merged_config = OmegaConf.create({})

        if self.architecture_config_obj is not None:
            # 1a: Check for explicit component_overrides in the architecture config
            if hasattr(self.architecture_config_obj, "component_overrides") and self.architecture_config_obj.component_overrides is not None:
                arch_overrides = self.architecture_config_obj.component_overrides
                merged_config = OmegaConf.merge(merged_config, arch_overrides)

            # 1b: Extract direct component definitions from architecture config
            potential_keys = ["encoder", "decoder", "head", "loss"]
            for key in potential_keys:
                if hasattr(self.architecture_config_obj, key):
                     val = getattr(self.architecture_config_obj, key)
                     if val is not None:
                         direct_overrides[key] = val
                elif isinstance(self.architecture_config_obj, dict) and key in self.architecture_config_obj:
                     direct_overrides[key] = self.architecture_config_obj[key]

        # Priority 2: Collect top-level component keys from cfg (Experiment/legacy overrides)
        # e.g. cfg.decoder from train_v2 -> _base/model -> dbnet.yaml
        top_level_overrides = {}
        for key in ["encoder", "decoder", "head", "loss"]:
            if hasattr(cfg, key):
                val = getattr(cfg, key)
                if val is not None:
                    top_level_overrides[key] = val

        # BUG_003 FIX: Filter conflicts before merging
        filtered_top_level = self._filter_architecture_conflicts(
            top_level_overrides,
            direct_overrides
        )

        # BUG_003 FIX: Corrected merge order
        # Legacy (filtered) merges BEFORE architecture direct_overrides
        if filtered_top_level:
            merged_config = OmegaConf.merge(merged_config, filtered_top_level)

        # Architecture direct components merge AFTER legacy (so they win)
        if direct_overrides:
            merged_config = OmegaConf.merge(merged_config, direct_overrides)

        # Priority 3: Check cfg.component_overrides (explicit user overrides, highest priority)
        # BUG_003 FIX: Also filter component_overrides for architecture conflicts
        # This handles legacy configs from _base/model.yaml -> model/architectures: dbnet
        if hasattr(cfg, "component_overrides") and cfg.component_overrides is not None:
            user_overrides = ensure_dict(cfg.component_overrides)
            if user_overrides:
                filtered_user_overrides = self._filter_architecture_conflicts(
                    user_overrides,
                    direct_overrides
                )
                if filtered_user_overrides:
                    merged_config = OmegaConf.merge(merged_config, filtered_user_overrides)

        # Use the fully merged config as the source of truth
        # Convert back to dict for processing
        overrides = ensure_dict(merged_config)

        component_configs: dict[str, Any] = {}
        if overrides is not None:
            for name in ("encoder", "decoder", "head", "loss"):
                if name not in overrides or overrides[name] is None:
                    continue

                section = ensure_dict(overrides[name])
                if not isinstance(section, dict):
                    raise TypeError(f"Component override for '{name}' must resolve to a mapping, got {type(section)!r}.")
                section_dict = cast(dict[str, Any], section)
                component_name_key = f"{name}_name"
                component_config_key = f"{name}_config"

                override_name: str | None = None
                override_params: dict[str, Any] | None

                if "name" in section_dict:
                    override_name = section_dict.get("name")
                    params = section_dict.get("params")
                    if params is None:
                        params = {k: v for k, v in section_dict.items() if k != "name"}
                    override_params = dict(params) if is_config(params) else {}
                else:
                    override_params = section_dict  # legacy format: plain parameter dict

                if override_name:
                    component_configs[component_name_key] = override_name

                component_configs[component_config_key] = override_params or {}
        return component_configs

    def _filter_architecture_conflicts(
        self,
        top_level_overrides: dict,
        architecture_overrides: dict
    ) -> dict:
        """Filter legacy config components that conflict with architecture.

        BUG_003 FIX: Remove components from top_level_overrides when the same
        component is defined in the architecture config with a different name.
        This prevents legacy defaults from polluting the merge.

        Args:
            top_level_overrides: Legacy component configs from cfg (e.g., from train_v2)
            architecture_overrides: Component configs from architecture definition

        Returns:
            Filtered top_level_overrides with conflicting components removed
        """
        if not architecture_overrides:
            return top_level_overrides

        filtered = dict(top_level_overrides)
        arch_components = {"encoder", "decoder", "head", "loss"}

        for component in arch_components:
            if component not in filtered or component not in architecture_overrides:
                continue

            # Get component names for comparison
            arch_cfg = architecture_overrides[component]
            legacy_cfg = filtered[component]

            arch_name = None
            legacy_name = None

            # Extract name from architecture config
            if hasattr(arch_cfg, "name"):
                arch_name = arch_cfg.name
            elif isinstance(arch_cfg, dict) and "name" in arch_cfg:
                arch_name = arch_cfg["name"]

            # Extract name from legacy config
            if hasattr(legacy_cfg, "name"):
                legacy_name = legacy_cfg.name
            elif isinstance(legacy_cfg, dict) and "name" in legacy_cfg:
                legacy_name = legacy_cfg["name"]

            # If both have names and they differ, filter the legacy one
            if arch_name and legacy_name and arch_name != legacy_name:
                logger.info(
                    f"BUG_003: Filtering legacy {component} ({legacy_name}) "
                    f"in favor of architecture {component} ({arch_name})"
                )
                del filtered[component]

        return filtered
