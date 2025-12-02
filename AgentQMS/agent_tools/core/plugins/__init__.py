"""
AgentQMS Plugin System

Provides plugin discovery, validation, and registry management for:
- Artifact Types
- Validators
- Context Bundles

Public API:
    from AgentQMS.agent_tools.core.plugins import (
        get_plugin_registry,
        get_plugin_loader,
        PluginLoader,
        PluginRegistry,
    )

    # Get the singleton registry (cached)
    registry = get_plugin_registry()

    # Access plugin data
    artifact_types = registry.get_artifact_types()
    validators = registry.get_validators()
    bundles = registry.get_context_bundles()

CLI Usage:
    python -m AgentQMS.agent_tools.core.plugins --list
    python -m AgentQMS.agent_tools.core.plugins --validate
    python -m AgentQMS.agent_tools.core.plugins --json
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Public exports from submodules
from .discovery import DiscoveredPlugin, PluginDiscovery
from .loader import PluginLoader
from .registry import PluginMetadata, PluginRegistry, PluginValidationError
from .snapshot import SnapshotWriter
from .validation import PluginValidator, SchemaValidationError

__all__ = [
    # Main API
    "get_plugin_registry",
    "get_plugin_loader",
    # Classes
    "PluginLoader",
    "PluginRegistry",
    "PluginDiscovery",
    "PluginValidator",
    "SnapshotWriter",
    # Data classes
    "PluginMetadata",
    "PluginValidationError",
    "DiscoveredPlugin",
    "SchemaValidationError",
]


# ---------------------------------------------------------------------------
# Singleton Management
# ---------------------------------------------------------------------------

_plugin_loader: Optional[PluginLoader] = None


def get_plugin_loader(project_root: Optional[Path] = None) -> PluginLoader:
    """
    Return a singleton plugin loader instance.

    Args:
        project_root: Optional project root. Only used on first call.

    Returns:
        The singleton PluginLoader instance.
    """
    global _plugin_loader
    if _plugin_loader is None:
        _plugin_loader = PluginLoader(project_root=project_root)
    return _plugin_loader


def get_plugin_registry(force: bool = False) -> PluginRegistry:
    """
    Get the plugin registry (convenience function).

    This loads all plugins from framework and project directories,
    validates them, and returns a merged registry.

    Args:
        force: If True, reload plugins from disk even if cached.

    Returns:
        PluginRegistry containing all discovered plugins.

    Example:
        registry = get_plugin_registry()
        
        # Get all artifact types
        types = registry.get_artifact_types()
        
        # Get a specific artifact type
        cr_type = registry.get_artifact_type("change_request")
        
        # Get validator extensions
        validators = registry.get_validators()
        
        # Check for errors
        if registry.has_errors():
            for error in registry.validation_errors:
                print(f"Error: {error.error_message}")
    """
    loader = get_plugin_loader()
    registry = loader.load(force=force)

    # Auto-write snapshot on first load
    if not force:
        try:
            from AgentQMS.agent_tools.utils.paths import get_project_root
            state_dir = get_project_root() / ".agentqms" / "state"
            writer = SnapshotWriter(state_dir)
            writer.write(registry, loader.get_discovery_paths())
        except Exception:
            pass  # Snapshot writing is non-critical

    return registry


def reset_plugin_loader() -> None:
    """
    Reset the singleton plugin loader.

    Useful for testing or when plugin files have changed.
    """
    global _plugin_loader
    _plugin_loader = None

