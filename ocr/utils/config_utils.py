"""Hydra configuration schemas and validation utilities for modular OCR architectures."""

from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig

from ocr.utils.path_utils import get_path_resolver

CONFIG_ROOT = get_path_resolver().config.config_dir
SCHEMA_DIR = CONFIG_ROOT / "schemas"

from typing import TypeGuard

from omegaconf import ListConfig, OmegaConf


def is_config(obj: Any) -> TypeGuard[dict | DictConfig]:
    """
    Check if object is a valid configuration object (dict or DictConfig).
    Use this instead of instance(obj, dict).
    """
    return isinstance(obj, (dict, DictConfig))


def ensure_dict(cfg: Any, resolve: bool = True) -> dict:
    """
    Recursively convert DictConfig/ListConfig to native dict/list.
    Safe to call on already-native objects.
    """
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=resolve)  # type: ignore
    if isinstance(cfg, dict):
        return {k: ensure_dict(v, resolve) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [ensure_dict(v, resolve) for v in cfg]
    return cfg



def load_config(config_name: str = "train", overrides: list[str] | None = None) -> DictConfig:
    """
    Load a Hydra config using OmegaConf to handle defaults merging.

    Simpler approach: @package directives are processed immediately during merging,
    not retrospectively after defaults are processed.

    Args:
        config_name: Name of the config file to load (e.g., 'train', 'predict')
        overrides: Optional list of Hydra overrides to apply

    Returns:
        Config DictConfig with all defaults merged
    """
    if overrides is None:
        overrides = []

    import os

    from omegaconf import OmegaConf

    config_dir = Path("configs").resolve()

    def resolve_config_path(path_spec: str) -> Path | None:
        """Resolve a Hydra config path specification to an actual file."""
        path_str = path_spec.replace("/", os.sep)

        # Strip leading slash
        if path_str.startswith(os.sep):
            path_str = path_str.lstrip(os.sep)

        # Handle configs/ prefix if it would result in duplication
        # config_dir is absolute path to 'configs' directory
        if path_str.startswith("configs" + os.sep):
            path_str = path_str[len("configs" + os.sep):]

        # Add extension if missing
        if not path_str.endswith(".yaml"):
            path_str += ".yaml"

        file_path = config_dir / path_str

        return file_path if file_path.exists() else None

    def extract_package_directive(file_path: Path) -> str | None:
        """Extract the @package directive from the YAML file."""
        try:
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("# @package"):
                        return line.replace("# @package", "").strip()
        except Exception:
            pass
        return None

    def nest_at_path(value: dict, path: str) -> dict:
        """Nest a dictionary at a dot-separated path."""
        if not path or path == "_global_":
            return value

        parts = path.split(".")
        result: dict[str, Any] = {}
        current = result

        for part in parts[:-1]:
            current[part] = {}
            current = current[part]

        current[parts[-1]] = value
        return result

    def process_config(file_path: Path, skip_own_package: bool = False) -> DictConfig:
        """
        Load a config file and process its defaults.

        Args:
            file_path: Path to the config file
            skip_own_package: If True, don't apply this file's @package directive
                             (used when called as a sub-config where parent handles packaging)
        """
        cfg = OmegaConf.load(str(file_path))

        if "defaults" not in cfg:
            # No defaults - just apply own @package if applicable
            if not skip_own_package:
                own_package = extract_package_directive(file_path)
                if own_package:
                    container = OmegaConf.to_container(cfg)
                    assert isinstance(container, dict)
                    nested = nest_at_path(container, own_package)
                    result = OmegaConf.create(nested)
                    assert isinstance(result, DictConfig)
                    return result
            assert isinstance(cfg, DictConfig)
            return cfg

        defaults = cfg.defaults
        merged_temp = OmegaConf.create({})
        assert isinstance(merged_temp, DictConfig)
        merged: DictConfig = merged_temp
        own_content = {k: v for k, v in cfg.items() if k != "defaults"}

        for default in defaults:
            if default == "_self_":
                merged_result = OmegaConf.merge(merged, own_content)
                assert isinstance(merged_result, DictConfig)
                merged = merged_result
                continue

            if is_config(default):
                # Dict-style: {'model': 'default'} means load model/default.yaml
                for key, value in default.items():
                    if key.startswith("/hydra"):
                        continue

                    config_path = resolve_config_path(f"{key}/{value}")
                    if config_path:
                        # Load the sub-config and apply ITS @package directive
                        sub_cfg = process_config(config_path, skip_own_package=False)
                        extract_package_directive(config_path)

                        # Remove defaults from sub_cfg
                        sub_cfg.pop("defaults", None)

                        # The sub_cfg from process_config already has @package applied
                        # So we just merge it
                        merged_result = OmegaConf.merge(merged, OmegaConf.to_container(sub_cfg))
                        assert isinstance(merged_result, DictConfig)
                        merged = merged_result
            else:
                # String-style
                default_str = str(default)
                if default_str.startswith("/hydra"):
                    continue

                config_path = resolve_config_path(default_str)
                if config_path:
                    sub_cfg = process_config(config_path, skip_own_package=False)

                    # Remove defaults from sub_cfg
                    sub_cfg.pop("defaults", None)

                    merged_result = OmegaConf.merge(merged, OmegaConf.to_container(sub_cfg))
                    assert isinstance(merged_result, DictConfig)
                    merged = merged_result

        # Apply _self_ at the end if not explicitly placed
        if "_self_" not in defaults:
            merged_result = OmegaConf.merge(merged, own_content)
            assert isinstance(merged_result, DictConfig)
            merged = merged_result

        # Apply this file's own @package directive (unless skipped)
        if not skip_own_package:
            own_package = extract_package_directive(file_path)
            if own_package:
                container = OmegaConf.to_container(merged)
                assert isinstance(container, dict)
                nested = nest_at_path(container, own_package)
                merged_result = OmegaConf.create(nested)
                assert isinstance(merged_result, DictConfig)
                merged = merged_result

        return merged

    try:
        config_file = resolve_config_path(config_name)
        if config_file is None:
            raise FileNotFoundError(f"Config file not found: {config_name}")

        cfg = process_config(config_file)

        # HACK: Fix double-nested model key
        # This happens because model/default.yaml and model/architectures/dbnet.yaml
        # both have @package model, causing double nesting
        if "model" in cfg and hasattr(cfg.model, "keys"):
            if "model" in cfg.model and "encoder" not in cfg.model:
                # We have cfg.model.model with encoder/decoder/head/loss
                # Also have cfg.model with optimizer and other top-level keys
                # Merge them properly
                model_content = cfg.model["model"]
                optimizer = cfg.model.get("optimizer")
                # Keep all keys from model_content and add optimizer if present
                merged_model = OmegaConf.merge(model_content, {"optimizer": optimizer} if optimizer else {})
                cfg.model = merged_model

        # Apply command-line overrides
        for override in overrides:
            parts = override.split("=", 1)
            if len(parts) == 2:
                OmegaConf.update(cfg, parts[0], parts[1], force_add=True)

        return cfg
    except Exception as e:
        raise RuntimeError(f"Failed to load config '{config_name}': {e}") from e


def validate_model_config(config: dict[str, Any]) -> list[str]:
    """Validate model configuration for modular architecture compatibility.

    Args:
        config: Model configuration dictionary

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Local import to avoid circular dependency
    from ocr.core import registry

    # Check required keys
    for key in ["encoder", "decoder", "head", "loss"]:
        if key not in config:
            errors.append(f"Missing required key: {key}")
        elif config[key] not in registry.list_encoders() + registry.list_decoders() + registry.list_heads() + registry.list_losses():
            errors.append(f"Unknown component name for {key}: {config[key]}")
    # Check optimizer and scheduler
    if "optimizer" not in config:
        errors.append("Missing optimizer configuration")
    return errors


def load_config_schema(schema_name: str) -> dict[str, Any]:
    """Load a configuration schema YAML file for validation or UI rendering."""
    schema_path = SCHEMA_DIR / f"{schema_name}.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with open(schema_path) as f:
        return yaml.safe_load(f)


def get_default_config() -> dict[str, Any]:
    """Load the default configuration for training/inference."""
    default_path = CONFIG_ROOT / "defaults.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")
    with open(default_path) as f:
        return yaml.safe_load(f)
