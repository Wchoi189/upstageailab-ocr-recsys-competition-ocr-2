"""
Plugin Discovery Module

Handles discovering plugin files from framework and project directories.
This module is responsible for finding YAML files and determining their source.

No validation or loading of plugin content is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from AgentQMS.tools.utils.paths import get_agent_tools_dir, get_project_root


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
    1. Framework plugins: {framework_root}/.agentqms/plugins/
    2. Project plugins: {project_root}/.agentqms/plugins/

    This class only finds files - it does not read or validate them.
    """

    def __init__(
        self,
        project_root: Path,
        framework_root: Path | None = None,
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
        # Framework plugins: {framework_root}/.agentqms/plugins/
        self.framework_plugins_dir = self.framework_root / ".agentqms" / "plugins"

        # Project plugins: {project_root}/.agentqms/plugins/
        self.project_plugins_dir = self.project_root / ".agentqms" / "plugins"

    def discover_all(self) -> list[DiscoveredPlugin]:
        """
        Discover all plugin files from all registered sources.

        Returns:
            List of DiscoveredPlugin objects, framework plugins first.
        """
        plugins: list[DiscoveredPlugin] = []

        # Framework plugins first (can be overridden by project)
        if self.framework_plugins_dir.exists():
            plugins.extend(self._discover_from_directory(self.framework_plugins_dir, source="framework"))

        # Project plugins second (override framework)
        if self.project_plugins_dir.exists():
            plugins.extend(self._discover_from_directory(self.project_plugins_dir, source="project"))

        return plugins

    def discover_by_type(self) -> dict[str, list[DiscoveredPlugin]]:
        """
        Discover plugins grouped by type.

        Returns:
            Dictionary mapping plugin types to lists of DiscoveredPlugin.
        """
        all_plugins = self.discover_all()

        grouped: dict[str, list[DiscoveredPlugin]] = {
            "artifact_type": [],
            "validators": [],
            "context_bundle": [],
        }

        for plugin in all_plugins:
            if plugin.plugin_type in grouped:
                grouped[plugin.plugin_type].append(plugin)

        return grouped

    def _discover_from_directory(self, base_dir: Path, source: str) -> list[DiscoveredPlugin]:
        """
        Discover plugins from a single base directory.

        Args:
            base_dir: Base plugins directory to scan.
            source: Source identifier ('framework' or 'project').

        Returns:
            List of DiscoveredPlugin objects found in this directory.
        """
        plugins: list[DiscoveredPlugin] = []

        # Artifact types: base_dir/artifact_types/**/*.yaml
        artifact_types_dir = base_dir / "artifact_types"
        if artifact_types_dir.exists():
            for yaml_file in sorted(artifact_types_dir.rglob("*.yaml")):
                plugins.append(
                    DiscoveredPlugin(
                        path=yaml_file,
                        plugin_type="artifact_type",
                        source=source,
                    )
                )

        # Validators: base_dir/validators.yaml
        validators_file = base_dir / "validators.yaml"
        if validators_file.exists():
            plugins.append(
                DiscoveredPlugin(
                    path=validators_file,
                    plugin_type="validators",
                    source=source,
                )
            )

        # Context bundles: base_dir/context_bundles/**/*.yaml
        context_bundles_dir = base_dir / "context_bundles"
        if context_bundles_dir.exists():
            for yaml_file in sorted(context_bundles_dir.rglob("*.yaml")):
                plugins.append(
                    DiscoveredPlugin(
                        path=yaml_file,
                        plugin_type="context_bundle",
                        source=source,
                    )
                )

        return plugins

    def get_discovery_paths(self) -> dict[str, str]:
        """
        Return the discovery paths for debugging/logging.

        Returns relative paths from project root for portability across
        different environments (local development, CI/CD, deployments).

        Returns:
            Dictionary with relative paths to framework and project plugin dirs.
        """
        try:
            # Use relative paths for portability
            framework_rel = self.framework_plugins_dir.relative_to(self.project_root)
            project_rel = self.project_plugins_dir.relative_to(self.project_root)
        except ValueError:
            # Fallback to absolute if paths don't share common root
            framework_rel = self.framework_plugins_dir
            project_rel = self.project_plugins_dir

        return {
            "framework": str(framework_rel),
            "project": str(project_rel),
        }

    def discover_tools(self) -> dict[str, list[Path]]:
        """Discover executable tools from the AgentQMS/tools directory."""
        return discover_tools(get_agent_tools_dir())


def discover_tools(tools_root: Path) -> dict[str, list[Path]]:
    """Discover executable tools from the AgentQMS/tools directory."""
    if not tools_root.exists():
        return {}

    categories: dict[str, list[Path]] = {}
    for category_dir in sorted(tools_root.iterdir()):
        if not category_dir.is_dir() or category_dir.name == "__pycache__":
            continue

        scripts = sorted(
            script
            for script in category_dir.glob("*.py")
            if script.name != "__init__.py"
        )
        if scripts:
            categories[category_dir.name] = scripts

    return categories


def _print_tool_catalog(tools_map: dict[str, list[Path]], project_root: Path) -> None:
    print("🔍 Available Agent Tools:")
    print()
    print("📁 Architecture:")
    print("   Implementation Layer: AgentQMS/tools/ (canonical)")
    print("   Agent Interface: AgentQMS/bin/")
    print()

    category_descriptions = {
        "core": "Essential automation tools (artifact creation, context bundles)",
        "compliance": "Compliance and validation tools",
        "documentation": "Documentation management tools",
        "utils": "Helper functions and utilities",
        "audit": "Audit framework tools",
        "maintenance": "System maintenance and cleanup tasks",
        "multi_agent": "Multi-agent coordination and sub-agent management",
    }

    if not tools_map:
        print("   (no tools found in AgentQMS/tools/)")
        return

    for category, tools in tools_map.items():
        desc = category_descriptions.get(category, "")
        print(f"📁 {category.upper()}: {desc}")

        for tool in tools:
            try:
                rel_path = tool.relative_to(project_root)
                print(f"   uv run python {rel_path.as_posix()}")
            except ValueError:
                print(f"   uv run python {tool.as_posix()}")
        print()

    print("💡 Usage:")
    print("   For agents: cd AgentQMS/bin/ && make help")
    print("   Direct CLI: uv run python AgentQMS/tools/<category>/<tool>.py")
    print("   See README.md for detailed usage information")
    print()


def main() -> None:
    tools_root = get_agent_tools_dir()
    project_root = get_project_root()
    tools_map = discover_tools(tools_root)
    _print_tool_catalog(tools_map, project_root)


if __name__ == "__main__":
    main()

