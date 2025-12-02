"""
Plugin Loader Module

Orchestrates plugin discovery, loading, validation, and merging.
This is the main entry point for loading plugins into a registry.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from .discovery import DiscoveredPlugin, PluginDiscovery
from .registry import PluginMetadata, PluginRegistry, PluginValidationError
from .validation import PluginValidator


class PluginLoader:
    """
    Loads and merges plugins from framework and project sources.

    This class orchestrates:
    1. Discovery - Finding plugin files
    2. Loading - Reading YAML content
    3. Validation - Checking against schemas
    4. Merging - Combining into registry

    Usage:
        loader = PluginLoader(project_root)
        registry = loader.load()
    """

    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the plugin loader.

        Args:
            project_root: Project root directory. If None, auto-detected.
        """
        if project_root is None:
            from AgentQMS.agent_tools.utils.paths import get_project_root
            project_root = get_project_root()

        self.project_root = project_root
        self.framework_root = project_root / "AgentQMS"

        # Initialize components
        self.discovery = PluginDiscovery(
            project_root=project_root,
            framework_root=self.framework_root,
        )
        self.validator = PluginValidator(
            schemas_dir=self.framework_root / "conventions" / "schemas"
        )

        # Cached registry
        self._registry: Optional[PluginRegistry] = None

    def load(self, force: bool = False) -> PluginRegistry:
        """
        Load and merge all plugins into a registry.

        Args:
            force: If True, reload even if already cached.

        Returns:
            PluginRegistry containing all discovered plugins.
        """
        if self._registry is not None and not force:
            return self._registry

        registry = PluginRegistry(
            loaded_at=datetime.now(timezone.utc).isoformat()
        )

        # Discover all plugins
        discovered = self.discovery.discover_by_type()

        # Load each type
        self._load_artifact_types(registry, discovered.get("artifact_type", []))
        self._load_validators(registry, discovered.get("validators", []))
        self._load_context_bundles(registry, discovered.get("context_bundle", []))

        self._registry = registry
        return registry

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file and return dictionary."""
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML format (expected dict): {path}")
        return data

    def _load_artifact_types(
        self, registry: PluginRegistry, plugins: list[DiscoveredPlugin]
    ) -> None:
        """Load and merge artifact type plugins."""
        for plugin in plugins:
            try:
                data = self._load_yaml(plugin.path)

                # Validate
                errors = self.validator.validate(data, "artifact_type")
                if errors:
                    for error in errors:
                        registry.add_validation_error(PluginValidationError(
                            plugin_path=str(plugin.path),
                            plugin_type="artifact_type",
                            error_message=error,
                            schema_path=str(self.validator.get_schema_path("artifact_type")),
                        ))
                    continue

                # Extract name
                name = data.get("name", plugin.path.stem)

                # Handle override logic
                if self._should_override(registry, name, "artifact_type", plugin.source):
                    registry.add_artifact_type(
                        name=name,
                        data=data,
                        metadata=PluginMetadata(
                            name=name,
                            version=data.get("version", "1.0"),
                            source=plugin.source,
                            path=str(plugin.path),
                            plugin_type="artifact_type",
                            scope=data.get("scope", "project"),
                            description=data.get("description", ""),
                        ),
                    )

            except Exception as e:
                registry.add_validation_error(PluginValidationError(
                    plugin_path=str(plugin.path),
                    plugin_type="artifact_type",
                    error_message=str(e),
                ))

    def _load_validators(
        self, registry: PluginRegistry, plugins: list[DiscoveredPlugin]
    ) -> None:
        """Load and merge validator plugins."""
        merged: Dict[str, Any] = {
            "prefixes": {},
            "types": [],
            "categories": [],
            "statuses": [],
            "rules": {},
            "custom_validators": [],
            "disabled_validators": [],
        }

        for plugin in plugins:
            try:
                data = self._load_yaml(plugin.path)

                # Validate
                errors = self.validator.validate(data, "validators")
                if errors:
                    for error in errors:
                        registry.add_validation_error(PluginValidationError(
                            plugin_path=str(plugin.path),
                            plugin_type="validators",
                            error_message=error,
                            schema_path=str(self.validator.get_schema_path("validators")),
                        ))
                    continue

                # Merge
                self._merge_validators(merged, data)

                # Track metadata
                registry.metadata.append(PluginMetadata(
                    name="validators",
                    version=data.get("version", "1.0"),
                    source=plugin.source,
                    path=str(plugin.path),
                    plugin_type="validator",
                    description=data.get("description", ""),
                ))

            except Exception as e:
                registry.add_validation_error(PluginValidationError(
                    plugin_path=str(plugin.path),
                    plugin_type="validators",
                    error_message=str(e),
                ))

        # Only add if we have meaningful data
        if any(merged[k] for k in ["prefixes", "types", "categories", "statuses"]):
            registry.validators = merged

    def _load_context_bundles(
        self, registry: PluginRegistry, plugins: list[DiscoveredPlugin]
    ) -> None:
        """Load and merge context bundle plugins."""
        for plugin in plugins:
            try:
                data = self._load_yaml(plugin.path)

                # Validate
                errors = self.validator.validate(data, "context_bundle")
                if errors:
                    for error in errors:
                        registry.add_validation_error(PluginValidationError(
                            plugin_path=str(plugin.path),
                            plugin_type="context_bundle",
                            error_message=error,
                            schema_path=str(self.validator.get_schema_path("context_bundle")),
                        ))
                    continue

                # Extract name
                name = data.get("name", plugin.path.stem)

                # Handle override logic
                if self._should_override(registry, name, "context_bundle", plugin.source):
                    registry.add_context_bundle(
                        name=name,
                        data=data,
                        metadata=PluginMetadata(
                            name=name,
                            version=data.get("version", "1.0"),
                            source=plugin.source,
                            path=str(plugin.path),
                            plugin_type="context_bundle",
                            scope=data.get("scope", "project"),
                            description=data.get("description", ""),
                        ),
                    )

            except Exception as e:
                registry.add_validation_error(PluginValidationError(
                    plugin_path=str(plugin.path),
                    plugin_type="context_bundle",
                    error_message=str(e),
                ))

    def _should_override(
        self,
        registry: PluginRegistry,
        name: str,
        plugin_type: str,
        new_source: str,
    ) -> bool:
        """
        Determine if a new plugin should override an existing one.

        Override rules:
        - Project always overrides framework
        - Same source: later file wins
        - Framework cannot override project
        """
        existing = registry.get_metadata_for_plugin(name, plugin_type)

        if existing is None:
            return True  # No existing plugin

        if new_source == "project" and existing.source == "framework":
            return True  # Project overrides framework

        if new_source == existing.source:
            return True  # Same source, later wins

        return False  # Framework cannot override project

    def _merge_validators(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> None:
        """Merge validator configurations (mutates base)."""
        # Merge prefixes (override wins)
        if "prefixes" in override:
            base["prefixes"].update(override["prefixes"])

        # Merge lists (unique values, sorted)
        for key in ["types", "categories", "statuses"]:
            if key in override:
                existing = set(base.get(key, []))
                existing.update(override.get(key, []))
                base[key] = sorted(existing)

        # Merge rules (override wins)
        if "rules" in override:
            base.setdefault("rules", {}).update(override["rules"])

        # Append custom validators
        if "custom_validators" in override:
            base.setdefault("custom_validators", []).extend(
                override["custom_validators"]
            )

        # Append disabled validators
        if "disabled_validators" in override:
            existing = set(base.get("disabled_validators", []))
            existing.update(override.get("disabled_validators", []))
            base["disabled_validators"] = sorted(existing)

    def get_discovery_paths(self) -> Dict[str, str]:
        """Get the discovery paths used by this loader."""
        return self.discovery.get_discovery_paths()

