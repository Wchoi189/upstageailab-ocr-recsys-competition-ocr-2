"""Configuration loading and validation service.

Handles loading YAML configs, schema validation, and config inheritance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import jsonschema
import streamlit as st
import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate YAML configurations with schema support."""

    def __init__(self, schema_dir: Path | str):
        """Initialize config loader.

        Args:
            schema_dir: Directory containing validation schemas
        """
        self.schema_dir = Path(schema_dir)

    def load_config(self, config_path: Path | str, schema_name: str | None = None, validate: bool = True) -> dict[str, Any]:
        """Load YAML config and optionally validate against schema.

        Args:
            config_path: Path to YAML config file
            schema_name: Name of validation schema (without .yaml extension)
            validate: Whether to perform schema validation

        Returns:
            Validated configuration dictionary

        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails
            jsonschema.ValidationError: If validation fails
        """
        config_path = Path(config_path)

        logger.info(f"Loading config from {config_path}")

        # Load YAML
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Empty config file: {config_path}")

        # Handle inheritance
        if "inherit_from" in config:
            parent_path = config_path.parent / config["inherit_from"]
            logger.info(f"Config inherits from {parent_path}")
            parent_config = self.load_config(parent_path, schema_name=None, validate=False)
            config = self._merge_configs(parent_config, config)

        # Validate if schema provided
        if validate and schema_name:
            self._validate_config(config, schema_name)

        logger.info(f"Successfully loaded config from {config_path}")
        return config

    def _validate_config(self, config: dict, schema_name: str) -> None:
        """Validate config against JSON schema.

        Args:
            config: Configuration dictionary
            schema_name: Name of schema file (without .yaml)

        Raises:
            jsonschema.ValidationError: If validation fails
            FileNotFoundError: If schema file not found
        """
        schema_path = self.schema_dir / f"{schema_name}.yaml"

        if not schema_path.exists():
            logger.warning(f"Schema file not found: {schema_path}, skipping validation")
            return

        logger.debug(f"Validating config against {schema_path}")

        with open(schema_path) as f:
            schema = yaml.safe_load(f)

        try:
            # Validate using jsonschema
            jsonschema.validate(config, schema)
            logger.debug("Config validation passed")

            # Apply custom validation rules if present
            if "custom_validation" in schema:
                self._apply_custom_validation(config, schema["custom_validation"])

        except jsonschema.ValidationError as e:
            logger.error(f"Config validation failed: {e.message}")
            logger.error(f"Failed at path: {list(e.path)}")
            raise

    def _apply_custom_validation(self, config: dict, rules: list[dict]) -> None:
        """Apply custom validation rules.

        Args:
            config: Configuration dictionary
            rules: List of custom validation rules

        Raises:
            ValueError: If custom validation fails
        """
        for rule in rules:
            rule_name = rule.get("rule", "unknown")
            condition = rule.get("condition", {})
            requires = rule.get("requires", {})
            error_message = rule.get("error_message", f"Custom validation failed: {rule_name}")

            # Check condition
            if self._check_condition(config, condition):
                # Check requirement
                if not self._check_condition(config, requires):
                    raise ValueError(error_message)

    def _check_condition(self, config: dict, condition: dict) -> bool:
        """Check if a condition is met in the config.

        Args:
            config: Configuration dictionary
            condition: Condition specification

        Returns:
            True if condition is met, False otherwise
        """
        if not condition:
            return True

        path = condition.get("path", "")
        value = self._get_nested_value(config, path)

        if "equals" in condition:
            return value == condition["equals"]
        if "greater_than" in condition:
            return value > condition["greater_than"]
        if "less_than" in condition:
            return value < condition["less_than"]
        if "exists" in condition:
            return value is not None

        return True

    def _get_nested_value(self, config: dict, path: str) -> Any:
        """Get nested value from config using dot notation.

        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., "background_removal.enable")

        Returns:
            Value at path, or None if not found
        """
        keys = path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _merge_configs(self, base: dict, override: dict) -> dict:
        """Recursively merge override into base config.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key == "inherit_from":
                continue  # Don't include inheritance directive

            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result


# Global loader instance
_loader: ConfigLoader | None = None


def _get_loader() -> ConfigLoader:
    """Get or create global config loader instance.

    Returns:
        ConfigLoader instance
    """
    global _loader
    if _loader is None:
        _loader = ConfigLoader(Path("configs/schemas"))
    return _loader


@st.cache_data(show_spinner=False, ttl=3600)
def load_unified_config(config_name: str = "unified_app") -> dict[str, Any]:
    """Load and cache unified app configuration.

    Args:
        config_name: Name of config file (without .yaml)

    Returns:
        Loaded and validated configuration

    Raises:
        FileNotFoundError: If config file not found
        ValidationError: If config validation fails
    """
    loader = _get_loader()
    config_path = Path(f"configs/ui/{config_name}.yaml")

    try:
        return loader.load_config(config_path, schema_name=None, validate=False)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


@st.cache_data(show_spinner=False, ttl=3600)
def load_mode_config(mode_id: str, validate: bool = True) -> dict[str, Any]:
    """Load mode-specific configuration.

    Args:
        mode_id: Mode identifier ('preprocessing', 'inference', 'comparison')
        validate: Whether to validate against schema

    Returns:
        Mode configuration dictionary

    Raises:
        FileNotFoundError: If mode config not found
        ValidationError: If validation fails
    """
    loader = _get_loader()
    config_path = Path(f"configs/ui/modes/{mode_id}.yaml")

    # Determine schema name
    schema_name = f"{mode_id}_schema" if validate else None

    try:
        return loader.load_config(config_path, schema_name=schema_name, validate=validate)
    except Exception as e:
        logger.error(f"Failed to load mode config for {mode_id}: {e}")
        raise
