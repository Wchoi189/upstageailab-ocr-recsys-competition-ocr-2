#!/usr/bin/env python3
"""
Plugin Registry for AgentQMS Framework

Loads plugin definitions from .agentqms/plugins/ directory and provides
a unified interface for accessing custom artifact types, validators, and context bundles.

This module implements the PluginRegistry class that is referenced throughout the
framework (validate_artifacts.py, context_bundle.py, etc.) but was missing from
the codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from AgentQMS.agent_tools.utils.paths import get_project_root
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


class PluginRegistry:
    """
    Loads and manages plugins from .agentqms/plugins/ directory.

    Supports:
    - Custom artifact types (artifact_types/*.yaml)
    - Custom validators (validators.yaml)
    - Custom context bundles (context_bundles/*.yaml)
    """

    def __init__(self, project_root: Path | None = None):
        """Initialize plugin registry.

        Args:
            project_root: Project root path (auto-detected if not provided)
        """
        if project_root is None:
            project_root = get_project_root()

        self.project_root = Path(project_root)
        self.plugins_dir = self.project_root / ".agentqms" / "plugins"

        self._validators: dict[str, Any] = {}
        self._artifact_types: dict[str, Any] = {}
        self._context_bundles: dict[str, Any] = {}

        # Load all plugins on initialization
        self._load_validators()
        self._load_artifact_types()
        self._load_context_bundles()

    def _load_validators(self) -> None:
        """Load validator definitions from .agentqms/plugins/validators.yaml"""
        validators_path = self.plugins_dir / "validators.yaml"
        if validators_path.exists():
            try:
                with open(validators_path, encoding="utf-8") as f:
                    self._validators = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"⚠️  Warning: Failed to load validators.yaml: {e}")
                self._validators = {}

    def _load_artifact_types(self) -> None:
        """Load artifact type definitions from .agentqms/plugins/artifact_types/"""
        artifact_types_dir = self.plugins_dir / "artifact_types"
        if not artifact_types_dir.exists():
            return

        for yaml_file in artifact_types_dir.glob("*.yaml"):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    artifact_def = yaml.safe_load(f) or {}
                    # Use filename (without .yaml) as the type key
                    type_name = yaml_file.stem
                    self._artifact_types[type_name] = artifact_def
            except Exception as e:
                print(f"⚠️  Warning: Failed to load artifact type {yaml_file.name}: {e}")

    def _load_context_bundles(self) -> None:
        """Load context bundle definitions from .agentqms/plugins/context_bundles/"""
        bundles_dir = self.plugins_dir / "context_bundles"
        if not bundles_dir.exists():
            return

        for yaml_file in bundles_dir.glob("*.yaml"):
            try:
                with open(yaml_file, encoding="utf-8") as f:
                    bundle_def = yaml.safe_load(f) or {}
                    # Use filename (without .yaml) as the bundle key
                    bundle_name = yaml_file.stem
                    self._context_bundles[bundle_name] = bundle_def
            except Exception as e:
                print(f"⚠️  Warning: Failed to load context bundle {yaml_file.name}: {e}")

    def get_validators(self) -> dict[str, Any]:
        """Get validator configuration.

        Returns:
            Dictionary containing:
            - prefixes: Custom artifact type prefixes
            - types: Custom artifact types
            - categories: Custom artifact categories
            - statuses: Custom artifact statuses
        """
        return self._validators

    def get_artifact_types(self) -> dict[str, Any]:
        """Get all registered artifact type definitions.

        Returns:
            Dictionary mapping type names to their definitions.
        """
        return self._artifact_types

    def get_artifact_type(self, type_name: str) -> dict[str, Any] | None:
        """Get a specific artifact type definition.

        Args:
            type_name: Name of the artifact type

        Returns:
            Artifact type definition or None if not found
        """
        return self._artifact_types.get(type_name)

    def get_context_bundles(self) -> dict[str, Any]:
        """Get all registered context bundles.

        Returns:
            Dictionary mapping bundle names to their definitions.
        """
        return self._context_bundles

    def get_context_bundle(self, bundle_name: str) -> dict[str, Any] | None:
        """Get a specific context bundle definition.

        Args:
            bundle_name: Name of the context bundle

        Returns:
            Context bundle definition or None if not found
        """
        return self._context_bundles.get(bundle_name)

    def has_artifact_type(self, type_name: str) -> bool:
        """Check if an artifact type is registered.

        Args:
            type_name: Name of the artifact type

        Returns:
            True if artifact type is registered
        """
        return type_name in self._artifact_types

    def has_context_bundle(self, bundle_name: str) -> bool:
        """Check if a context bundle is registered.

        Args:
            bundle_name: Name of the context bundle

        Returns:
            True if context bundle is registered
        """
        return bundle_name in self._context_bundles

    def list_artifact_types(self) -> list[str]:
        """List all registered artifact type names.

        Returns:
            List of artifact type names
        """
        return sorted(self._artifact_types.keys())

    def list_context_bundles(self) -> list[str]:
        """List all registered context bundle names.

        Returns:
            List of context bundle names
        """
        return sorted(self._context_bundles.keys())

    def reload(self) -> None:
        """Reload all plugins from disk.

        Useful if plugins directory was modified externally.
        """
        self._validators = {}
        self._artifact_types = {}
        self._context_bundles = {}

        self._load_validators()
        self._load_artifact_types()
        self._load_context_bundles()


# Global registry singleton
_registry: PluginRegistry | None = None


def get_plugin_registry() -> PluginRegistry:
    """Get or create the global plugin registry.

    Returns:
        PluginRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def reset_plugin_registry() -> None:
    """Reset the global plugin registry (useful for testing)."""
    global _registry
    _registry = None


__all__ = [
    "PluginRegistry",
    "get_plugin_registry",
    "reset_plugin_registry",
]
