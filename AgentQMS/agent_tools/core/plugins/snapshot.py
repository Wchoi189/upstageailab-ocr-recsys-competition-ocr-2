"""
Plugin Snapshot Module

Handles writing runtime snapshots of the plugin registry to disk.
This provides an audit trail and debugging aid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from .registry import PluginRegistry


class SnapshotWriter:
    """
    Writes plugin registry snapshots to disk.

    The snapshot captures:
    - Discovery paths used
    - Plugins loaded by type
    - Full plugin metadata
    - Any validation errors
    - Complete resolved plugin data
    """

    def __init__(self, state_dir: Path):
        """
        Initialize the snapshot writer.

        Args:
            state_dir: Directory to write snapshots to.
        """
        self.state_dir = state_dir
        self.snapshot_path = state_dir / "plugins.yaml"

    def write(
        self,
        registry: PluginRegistry,
        discovery_paths: Dict[str, str],
    ) -> Path:
        """
        Write a snapshot of the plugin registry.

        Args:
            registry: The plugin registry to snapshot.
            discovery_paths: The discovery paths used.

        Returns:
            Path to the written snapshot file.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)

        snapshot = self._build_snapshot(registry, discovery_paths)

        with self.snapshot_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(snapshot, f, sort_keys=False, default_flow_style=False)

        return self.snapshot_path

    def _build_snapshot(
        self,
        registry: PluginRegistry,
        discovery_paths: Dict[str, str],
    ) -> Dict[str, Any]:
        """Build the snapshot dictionary."""
        return {
            "metadata": {
                "generated_at": registry.loaded_at,
                "generator": "AgentQMS PluginLoader",
                "schema_version": "1.0",
            },
            "discovery_paths": discovery_paths,
            "plugins_loaded": {
                "artifact_types": list(registry.artifact_types.keys()),
                "validators": bool(registry.validators),
                "context_bundles": list(registry.context_bundles.keys()),
            },
            "plugin_metadata": [m.to_dict() for m in registry.metadata],
            "validation_errors": [e.to_dict() for e in registry.validation_errors],
            "resolved": {
                "artifact_types": registry.artifact_types,
                "validators": registry.validators,
                "context_bundles": registry.context_bundles,
            },
        }

    def read(self) -> Dict[str, Any]:
        """
        Read the existing snapshot (if any).

        Returns:
            Snapshot dictionary or empty dict if not found.
        """
        if not self.snapshot_path.exists():
            return {}

        try:
            with self.snapshot_path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}

