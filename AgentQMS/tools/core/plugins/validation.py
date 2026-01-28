"""
Plugin Validation Module

Handles schema validation for plugin configurations.
This module validates plugin data dictionaries against JSON schemas
and enforces artifact type validation rules from centralized configuration.

No file I/O for plugins is performed here - only schema loading and validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

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

    def __init__(self, message: str, path: list[Any] | None = None):
        super().__init__(message)
        self.message = message
        self.path = path or []


class PluginValidator:
    """
    Validates plugin data against JSON schemas and artifact type rules.

    Schemas are loaded from the AgentQMS/standards/schemas directory.
    Artifact type validation rules are loaded from .agentqms/schemas/artifact_type_validation.yaml

    If jsonschema is not installed, schema validation is skipped.
    Artifact type validation always runs (naming, prohibited types, etc.)
    """

    # Mapping of plugin types to their schema files
    SCHEMA_MAP = {
        "artifact_type": "plugin_artifact_type.json",
        "validators": "plugin_validators.json",
        "context_bundle": "plugin_context_bundle.json",
    }

    def __init__(self, schemas_dir: Path | None = None, validation_rules_path: Path | None = None):
        """
        Initialize the validator.

        Args:
            schemas_dir: Directory containing JSON schema files.
                         If None, schema validation is disabled.
            validation_rules_path: Path to artifact_type_validation.yaml.
                                   If None, looks in .agentqms/schemas/
        """
        self.schemas_dir = schemas_dir
        self._schema_cache: dict[str, dict[str, Any]] = {}

        # Load artifact type validation rules
        self._validation_rules: dict[str, Any] = {}
        self._load_validation_rules(validation_rules_path)

    @property
    def is_available(self) -> bool:
        """Check if schema validation is available."""
        return JSONSCHEMA_AVAILABLE and self.schemas_dir is not None

    def _load_validation_rules(self, rules_path: Path | None = None) -> None:
        """
        Load artifact type validation rules from YAML.

        Args:
            rules_path: Path to artifact_type_validation.yaml.
                       If None, attempts to find it in .agentqms/schemas/
        """
        if rules_path is None:
            # Try to find it relative to common project structure
            candidates = [
                Path(".agentqms/schemas/artifact_type_validation.yaml"),
                Path("../.agentqms/schemas/artifact_type_validation.yaml"),
                Path("../../.agentqms/schemas/artifact_type_validation.yaml"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    rules_path = candidate
                    break

        if rules_path is None or not rules_path.exists():
            # No validation rules found - continue with empty rules
            return

        try:
            with rules_path.open("r", encoding="utf-8") as f:
                self._validation_rules = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError):
            # Failed to load rules - continue with empty rules
            pass

    def validate(self, plugin_data: dict[str, Any], plugin_type: str) -> list[str]:
        """
        Validate plugin data against its schema and artifact type rules.

        Args:
            plugin_data: Plugin configuration dictionary.
            plugin_type: Type of plugin ('artifact_type', 'validators', 'context_bundle').

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: list[str] = []

        # First, perform JSON schema validation
        if self.is_available:
            if plugin_type not in self.SCHEMA_MAP:
                errors.append(f"Unknown plugin type: {plugin_type}")
                return errors

            schema = self._load_schema(plugin_type)
            if schema is None:
                errors.append(f"Schema not found for plugin type: {plugin_type}")
                return errors

            try:
                validate(instance=plugin_data, schema=schema)
            except ValidationError as e:
                errors.append(f"{e.message} (path: {list(e.path)})")
            except Exception as e:
                errors.append(f"Validation error: {e}")

        # Second, perform artifact type-specific validation
        if plugin_type == "artifact_type":
            artifact_errors = self._validate_artifact_type(plugin_data)
            errors.extend(artifact_errors)

        return errors

    def validate_or_raise(self, plugin_data: dict[str, Any], plugin_type: str) -> None:
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

    def _load_schema(self, plugin_type: str) -> dict[str, Any] | None:
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
        except (OSError, json.JSONDecodeError):
            return None

    def get_schema(self, plugin_type: str) -> dict[str, Any] | None:
        """
        Get the schema for a plugin type (for inspection).

        Args:
            plugin_type: Type of plugin.

        Returns:
            Schema dictionary or None.
        """
        return self._load_schema(plugin_type)

    def get_schema_path(self, plugin_type: str) -> Path | None:
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

    def _validate_artifact_type(self, plugin_data: dict[str, Any]) -> list[str]:
        """
        Validate artifact type plugin against centralized rules.

        Phase 6.6: This method is now FULLY DATA-DRIVEN from artifact_type_validation.yaml.
        NO validation logic is hard-coded here - all rules come from the YAML schema.

        Args:
            plugin_data: Artifact type plugin data

        Returns:
            List of validation error messages
        """
        errors: list[str] = []

        if not self._validation_rules:
            return errors  # No rules loaded, skip validation

        plugin_name = plugin_data.get("name", "")
        canonical_types = self._validation_rules.get("canonical_types", {})
        prohibited = self._validation_rules.get("prohibited_types", [])

        self._check_prohibited_types(plugin_name, prohibited, errors)
        self._check_canonical_types(plugin_name, canonical_types, prohibited, errors)
        self._validate_plugin_metadata(plugin_data, errors)
        self._validate_plugin_template(plugin_data, errors)

        return errors

    def _check_prohibited_types(self, plugin_name: str, prohibited: list[dict[str, Any]], errors: list[str]) -> None:
        """Check if type is prohibited."""
        prohibited_names = [p.get("name") for p in prohibited]
        if plugin_name in prohibited_names:
            matching = next((p for p in prohibited if p.get("name") == plugin_name), None)
            if matching:
                use_instead = matching.get("use_instead", "unknown")
                reason = matching.get("reason", "")
                errors.append(
                    f"Prohibited artifact type '{plugin_name}'. "
                    f"Use '{use_instead}' instead. Reason: {reason}"
                )

    def _check_canonical_types(
        self,
        plugin_name: str,
        canonical_types: dict[str, Any],
        prohibited: list[dict[str, Any]],
        errors: list[str]
    ) -> None:
        """Check if type is canonical or alias."""
        prohibited_names = [p.get("name") for p in prohibited]
        if plugin_name not in canonical_types:
            # Check aliases
            found_alias = False
            for canonical_name, canonical_def in canonical_types.items():
                aliases = canonical_def.get("aliases", [])
                if plugin_name in aliases:
                    found_alias = True
                    errors.append(
                        f"Plugin name '{plugin_name}' is an alias. "
                        f"Use canonical name '{canonical_name}' instead."
                    )
                    break

            if not found_alias and plugin_name not in prohibited_names:
                # Unknown type (not canonical, not alias, not prohibited)
                valid_types = ", ".join(canonical_types.keys())
                errors.append(
                    f"Unknown artifact type '{plugin_name}'. "
                    f"Valid types: {valid_types}"
                )

    def _validate_plugin_metadata(self, plugin_data: dict[str, Any], errors: list[str]) -> None:
        """Validate metadata section."""
        metadata = plugin_data.get("metadata", {})
        if not metadata:
            errors.append("Missing 'metadata' section")
            return

        # Get required metadata fields from validation rules (not hard-coded)
        required_metadata_fields = self._validation_rules.get(
            "required_plugin_metadata_fields",
            ["filename_pattern", "directory", "frontmatter"]  # Fallback if not in YAML
        )
        for field in required_metadata_fields:
            if field not in metadata:
                errors.append(f"Missing required metadata field: {field}")

        # Phase 6.6: Frontmatter validation (data-driven from YAML)
        frontmatter = metadata.get("frontmatter", {})
        if frontmatter:
            required_frontmatter_fields = self._validation_rules.get(
                "required_frontmatter_fields",
                ["ads_version", "type", "category", "status", "version", "tags"]  # Fallback
            )
            for field in required_frontmatter_fields:
                if field not in frontmatter:
                    errors.append(f"Missing required frontmatter field: {field}")

    def _validate_plugin_template(self, plugin_data: dict[str, Any], errors: list[str]) -> None:
        """Validate template section."""
        template = plugin_data.get("template", "")
        if not template or not isinstance(template, str):
            errors.append("Missing or invalid 'template' field")

    def get_canonical_types(self) -> dict[str, Any]:
        """Get canonical artifact types from validation rules."""
        return self._validation_rules.get("canonical_types", {})

    def get_prohibited_types(self) -> list[dict[str, Any]]:
        """Get prohibited artifact types from validation rules."""
        return self._validation_rules.get("prohibited_types", [])
