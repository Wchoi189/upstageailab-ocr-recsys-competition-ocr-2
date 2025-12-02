"""
Plugin Discovery Module

Handles discovering plugin files from framework and project directories.
This module is responsible for finding YAML files and determining their source.

No validation or loading of plugin content is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DiscoveredPlugin:
    """Represents a discovered plugin file."""

    path: Path
    plugin_type: str  # 'artifact_type', 'validators', 'context_bundle'
    source: str  # 'framework' or 'project'

    def __str__(self) -> str:
        return f"{self.plugin_type}:{self.path.name} [{self.source}]"


class PluginDiscovery:
    """
    Discovers plugin files from registered directories.

    Discovery paths:
    1. Framework plugins: {framework_root}/conventions/plugins/
    2. Project plugins: {project_root}/.agentqms/plugins/

    This class only finds files - it does not read or validate them.
    """

    def __init__(
        self,
        project_root: Path,
        framework_root: Optional[Path] = None,
    ):
        """
        Initialize plugin discovery.

        Args:
            project_root: Project root directory.
            framework_root: Framework root directory. Defaults to {project_root}/AgentQMS.
        """
        self.project_root = project_root
        self.framework_root = framework_root or (project_root / "AgentQMS")

        # Standard plugin directories
        self.framework_plugins_dir = self.framework_root / "conventions" / "plugins"
        self.project_plugins_dir = self.project_root / ".agentqms" / "plugins"

    def discover_all(self) -> List[DiscoveredPlugin]:
        """
        Discover all plugin files from all registered sources.

        Returns:
            List of DiscoveredPlugin objects, framework plugins first.
        """
        plugins: List[DiscoveredPlugin] = []

        # Framework plugins first (can be overridden by project)
        if self.framework_plugins_dir.exists():
            plugins.extend(self._discover_from_directory(
                self.framework_plugins_dir, source="framework"
            ))

        # Project plugins second (override framework)
        if self.project_plugins_dir.exists():
            plugins.extend(self._discover_from_directory(
                self.project_plugins_dir, source="project"
            ))

        return plugins

    def discover_by_type(self) -> Dict[str, List[DiscoveredPlugin]]:
        """
        Discover plugins grouped by type.

        Returns:
            Dictionary mapping plugin types to lists of DiscoveredPlugin.
        """
        all_plugins = self.discover_all()

        grouped: Dict[str, List[DiscoveredPlugin]] = {
            "artifact_type": [],
            "validators": [],
            "context_bundle": [],
        }

        for plugin in all_plugins:
            if plugin.plugin_type in grouped:
                grouped[plugin.plugin_type].append(plugin)

        return grouped

    def _discover_from_directory(
        self, base_dir: Path, source: str
    ) -> List[DiscoveredPlugin]:
        """
        Discover plugins from a single base directory.

        Args:
            base_dir: Base plugins directory to scan.
            source: Source identifier ('framework' or 'project').

        Returns:
            List of DiscoveredPlugin objects found in this directory.
        """
        plugins: List[DiscoveredPlugin] = []

        # Artifact types: base_dir/artifact_types/*.yaml
        artifact_types_dir = base_dir / "artifact_types"
        if artifact_types_dir.exists():
            for yaml_file in sorted(artifact_types_dir.glob("*.yaml")):
                plugins.append(DiscoveredPlugin(
                    path=yaml_file,
                    plugin_type="artifact_type",
                    source=source,
                ))

        # Validators: base_dir/validators.yaml
        validators_file = base_dir / "validators.yaml"
        if validators_file.exists():
            plugins.append(DiscoveredPlugin(
                path=validators_file,
                plugin_type="validators",
                source=source,
            ))

        # Context bundles: base_dir/context_bundles/*.yaml
        context_bundles_dir = base_dir / "context_bundles"
        if context_bundles_dir.exists():
            for yaml_file in sorted(context_bundles_dir.glob("*.yaml")):
                plugins.append(DiscoveredPlugin(
                    path=yaml_file,
                    plugin_type="context_bundle",
                    source=source,
                ))

        return plugins

    def get_discovery_paths(self) -> Dict[str, str]:
        """Return the discovery paths for debugging/logging."""
        return {
            "framework": str(self.framework_plugins_dir),
            "project": str(self.project_plugins_dir),
        }

