#!/usr/bin/env python3
"""
Plugin Loader Shim

This module is a backwards-compatibility shim that delegates to the
modular plugin system at AgentQMS.agent_tools.core.plugins.

For new code, import directly from the plugins package:

    from AgentQMS.agent_tools.core.plugins import (
        get_plugin_registry,
        get_plugin_loader,
        PluginLoader,
        PluginRegistry,
    )

This shim will be deprecated in a future version.
"""

from __future__ import annotations

# Re-export everything from the modular package
from AgentQMS.agent_tools.core.plugins import (
    DiscoveredPlugin,
    PluginDiscovery,
    PluginLoader,
    PluginMetadata,
    PluginRegistry,
    PluginValidationError,
    PluginValidator,
    SchemaValidationError,
    SnapshotWriter,
    get_plugin_loader,
    get_plugin_registry,
    reset_plugin_loader,
)

__all__ = [
    # Main API
    "get_plugin_registry",
    "get_plugin_loader",
    "reset_plugin_loader",
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


def main() -> int:
    """CLI entry point - delegates to plugins.cli."""
    from AgentQMS.agent_tools.core.plugins.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    raise SystemExit(main())
