"""
Plugin Registry Module

Defines the PluginRegistry and related data structures for storing
discovered and validated plugins.

This module contains pure data structures with no I/O dependencies,
making it easy to test in isolation.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PluginValidationError:
    """Represents a plugin validation error."""

    plugin_path: str
    plugin_type: str
    error_message: str
    schema_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.plugin_path,
            "type": self.plugin_type,
            "error": self.error_message,
            "schema_path": self.schema_path,
        }


@dataclass
class PluginMetadata:
    """Metadata about a loaded plugin."""

    name: str
    version: str
    source: str  # 'framework' or 'project'
    path: str
    plugin_type: str  # 'artifact_type', 'validator', 'context_bundle'
    scope: str = "project"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.plugin_type,
            "version": self.version,
            "source": self.source,
            "path": self.path,
        }


@dataclass
class PluginRegistry:
    """
    Central registry holding all discovered and validated plugins.

    Provides access to merged plugin configurations for:
    - Artifact types
    - Validators
    - Context bundles

    This is a pure data container with accessor methods.
    No I/O operations are performed by this class.
    """

    artifact_types: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    validators: Dict[str, Any] = field(default_factory=dict)
    context_bundles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: List[PluginMetadata] = field(default_factory=list)
    validation_errors: List[PluginValidationError] = field(default_factory=list)
    loaded_at: Optional[str] = None

    def get_artifact_types(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered artifact types."""
        return deepcopy(self.artifact_types)

    def get_artifact_type(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific artifact type by name."""
        return deepcopy(self.artifact_types.get(name))

    def get_validators(self) -> Dict[str, Any]:
        """Get merged validator configuration."""
        return deepcopy(self.validators)

    def get_context_bundles(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered context bundles."""
        return deepcopy(self.context_bundles)

    def get_context_bundle(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific context bundle by name."""
        return deepcopy(self.context_bundles.get(name))

    def has_errors(self) -> bool:
        """Check if any validation errors occurred during loading."""
        return len(self.validation_errors) > 0

    def get_plugin_names(self, plugin_type: str) -> List[str]:
        """Get names of all plugins of a given type."""
        if plugin_type == "artifact_type":
            return list(self.artifact_types.keys())
        elif plugin_type == "context_bundle":
            return list(self.context_bundles.keys())
        elif plugin_type == "validator":
            return ["validators"] if self.validators else []
        return []

    def add_artifact_type(
        self, name: str, data: Dict[str, Any], metadata: PluginMetadata
    ) -> None:
        """Add an artifact type to the registry."""
        self.artifact_types[name] = data
        self.metadata.append(metadata)

    def add_context_bundle(
        self, name: str, data: Dict[str, Any], metadata: PluginMetadata
    ) -> None:
        """Add a context bundle to the registry."""
        self.context_bundles[name] = data
        self.metadata.append(metadata)

    def add_validation_error(self, error: PluginValidationError) -> None:
        """Add a validation error to the registry."""
        self.validation_errors.append(error)

    def get_metadata_for_plugin(
        self, name: str, plugin_type: str
    ) -> Optional[PluginMetadata]:
        """Get metadata for a specific plugin."""
        for m in self.metadata:
            if m.name == name and m.plugin_type == plugin_type:
                return m
        return None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Return a summary dictionary for JSON serialization."""
        return {
            "artifact_types": list(self.artifact_types.keys()),
            "context_bundles": list(self.context_bundles.keys()),
            "validators": bool(self.validators),
            "errors": len(self.validation_errors),
        }

