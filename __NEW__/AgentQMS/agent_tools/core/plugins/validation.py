"""
Plugin Validation Module

Handles schema validation for plugin configurations.
This module validates plugin data dictionaries against JSON schemas.

No file I/O for plugins is performed here - only schema loading and validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional jsonschema import
try:
    import jsonschema
    from jsonschema import ValidationError, validate

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    jsonschema = None  # type: ignore
    ValidationError = Exception  # type: ignore
    validate = None  # type: ignore
    JSONSCHEMA_AVAILABLE = False


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""

    def __init__(self, message: str, path: List[Any] = None):
        super().__init__(message)
        self.message = message
        self.path = path or []


class PluginValidator:
    """
    Validates plugin data against JSON schemas.

    Schemas are loaded from the conventions/schemas directory.
    If jsonschema is not installed, validation is skipped.
    """

    # Mapping of plugin types to their schema files
    SCHEMA_MAP = {
        "artifact_type": "plugin_artifact_type.json",
        "validators": "plugin_validators.json",
        "context_bundle": "plugin_context_bundle.json",
    }

    def __init__(self, schemas_dir: Optional[Path] = None):
        """
        Initialize the validator.

        Args:
            schemas_dir: Directory containing JSON schema files.
                         If None, schema validation is disabled.
        """
        self.schemas_dir = schemas_dir
        self._schema_cache: Dict[str, Dict[str, Any]] = {}

    @property
    def is_available(self) -> bool:
        """Check if schema validation is available."""
        return JSONSCHEMA_AVAILABLE and self.schemas_dir is not None

    def validate(
        self, plugin_data: Dict[str, Any], plugin_type: str
    ) -> List[str]:
        """
        Validate plugin data against its schema.

        Args:
            plugin_data: Plugin configuration dictionary.
            plugin_type: Type of plugin ('artifact_type', 'validators', 'context_bundle').

        Returns:
            List of validation error messages (empty if valid).
        """
        if not self.is_available:
            return []  # Skip validation if not available

        if plugin_type not in self.SCHEMA_MAP:
            return [f"Unknown plugin type: {plugin_type}"]

        schema = self._load_schema(plugin_type)
        if schema is None:
            return [f"Schema not found for plugin type: {plugin_type}"]

        try:
            validate(instance=plugin_data, schema=schema)
            return []
        except ValidationError as e:
            return [f"{e.message} (path: {list(e.path)})"]
        except Exception as e:
            return [f"Validation error: {e}"]

    def validate_or_raise(
        self, plugin_data: Dict[str, Any], plugin_type: str
    ) -> None:
        """
        Validate plugin data and raise if invalid.

        Args:
            plugin_data: Plugin configuration dictionary.
            plugin_type: Type of plugin.

        Raises:
            SchemaValidationError: If validation fails.
        """
        errors = self.validate(plugin_data, plugin_type)
        if errors:
            raise SchemaValidationError("; ".join(errors))

    def _load_schema(self, plugin_type: str) -> Optional[Dict[str, Any]]:
        """
        Load and cache a schema file.

        Args:
            plugin_type: Type of plugin to get schema for.

        Returns:
            Schema dictionary or None if not found.
        """
        if plugin_type in self._schema_cache:
            return self._schema_cache[plugin_type]

        if self.schemas_dir is None:
            return None

        schema_file = self.SCHEMA_MAP.get(plugin_type)
        if not schema_file:
            return None

        schema_path = self.schemas_dir / schema_file
        if not schema_path.exists():
            return None

        try:
            with schema_path.open("r", encoding="utf-8") as f:
                schema = json.load(f)
            self._schema_cache[plugin_type] = schema
            return schema
        except (json.JSONDecodeError, IOError):
            return None

    def get_schema(self, plugin_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the schema for a plugin type (for inspection).

        Args:
            plugin_type: Type of plugin.

        Returns:
            Schema dictionary or None.
        """
        return self._load_schema(plugin_type)

    def get_schema_path(self, plugin_type: str) -> Optional[Path]:
        """
        Get the path to a schema file.

        Args:
            plugin_type: Type of plugin.

        Returns:
            Path to schema file or None.
        """
        if self.schemas_dir is None:
            return None

        schema_file = self.SCHEMA_MAP.get(plugin_type)
        if not schema_file:
            return None

        schema_path = self.schemas_dir / schema_file
        return schema_path if schema_path.exists() else None

