"""Framework configuration loader for AgentQMS.

This module consolidates configuration data from the framework defaults,
framework-level configuration, project-level overrides, and environment
variables. The loader exposes helpers the rest of the toolchain can reuse,
ensuring every component resolves paths the same way.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Iterable
import os

import yaml



_DEFAULT_CONFIG: Dict[str, Any] = {}


class ConfigLoader:
    """Central configuration loader with caching."""

    def __init__(self) -> None:
        self._config_cache: Optional[Dict[str, Any]] = None
        self.framework_root = self._detect_framework_root()
        self.project_root = self._detect_project_root(self.framework_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load(self, force: bool = False) -> Dict[str, Any]:
        """Load configuration following the precedence hierarchy."""
        if self._config_cache is not None and not force:
            return deepcopy(self._config_cache)

        # Prefer consolidated .agentqms/settings.yaml when present.
        settings_path = self.project_root / ".agentqms" / "settings.yaml"
        if settings_path.exists():
            config = self._load_yaml(settings_path)
            config = self._merge_config(config, self._load_environment_overrides())
            self._write_runtime_snapshot(config, settings_path=settings_path)
        else:
            config = deepcopy(_DEFAULT_CONFIG)
            config = self._merge_config(config, self._load_framework_defaults())
            config = self._merge_config(config, self._load_project_overrides())
            config = self._merge_config(config, self._load_environment_overrides())
            self._write_runtime_snapshot(config)

        self._config_cache = config
        return deepcopy(config)

    def get_path(self, key: str) -> Path:
        """Return a project-relative path resolved from configuration."""
        config = self.load()
        value = config.get("paths", {}).get(key)
        if not value:
            raise KeyError(f"No path configured for '{key}'")

        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return self.project_root / candidate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _detect_framework_root(self) -> Path:
        current = Path(__file__).resolve()
        for parent in (current,) + tuple(current.parents):
            if parent.name == "AgentQMS":
                return parent
        raise RuntimeError("Could not determine framework root. Is AgentQMS installed?")

    def _detect_project_root(self, framework_root: Path) -> Path:
        if framework_root.name == "AgentQMS":
            return framework_root.parent
        return framework_root

    def _load_framework_defaults(self) -> Dict[str, Any]:
        defaults_dir = self.framework_root / "config_defaults"
        config: Dict[str, Any] = {}
        yaml_files: Iterable[Path] = (
            defaults_dir / "framework.yaml",
            defaults_dir / "interface.yaml",
            defaults_dir / "paths.yaml",
        )
        for path in yaml_files:
            config = self._merge_yaml_if_exists(config, path)

        tool_mappings = defaults_dir / "tool_mappings.json"
        if tool_mappings.exists():
            with tool_mappings.open("r", encoding="utf-8") as handle:
                config["tool_mappings"] = json.load(handle)
        return config

    def _load_project_overrides(self) -> Dict[str, Any]:
        # Check for project-level config/ (used by consuming projects)
        config_dir = self.project_root / "config"
        config: Dict[str, Any] = {}

        if config_dir.exists():
            yaml_files: Iterable[Path] = (
                config_dir / "framework.yaml",
                config_dir / "interface.yaml",
                config_dir / "paths.yaml",
            )
            for path in yaml_files:
                config = self._merge_yaml_if_exists(config, path)

            config = self._merge_directory_overrides(config, config_dir / "environments")
            config = self._merge_directory_overrides(config, config_dir / "overrides")
        else:
            # Framework project's own config in .agentqms/project_config/
            # (avoids conflicts when framework is imported into projects with their own config/)
            framework_config_dir = self.project_root / ".agentqms" / "project_config"
            if framework_config_dir.exists():
                yaml_files: Iterable[Path] = (
                    framework_config_dir / "framework.yaml",
                    framework_config_dir / "interface.yaml",
                    framework_config_dir / "paths.yaml",
                )
                for path in yaml_files:
                    config = self._merge_yaml_if_exists(config, path)

                config = self._merge_directory_overrides(config, framework_config_dir / "environments")
                config = self._merge_directory_overrides(config, framework_config_dir / "overrides")

        return config

    def _load_environment_overrides(self) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}

        artifacts = os.getenv("AGENTQMS_PATHS_ARTIFACTS")
        docs = os.getenv("AGENTQMS_PATHS_DOCS")
        strict_mode = os.getenv("AGENTQMS_VALIDATION_STRICT_MODE")

        if artifacts:
            overrides.setdefault("paths", {})["artifacts"] = artifacts
        if docs:
            overrides.setdefault("paths", {})["docs"] = docs
        if strict_mode is not None:
            overrides.setdefault("validation", {})["strict_mode"] = (
                strict_mode.lower() == "true"
            )
        return overrides

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid configuration format: {path}")
        return data

    def _merge_yaml_if_exists(self, base: Dict[str, Any], path: Path) -> Dict[str, Any]:
        if not path.exists():
            return base
        return self._merge_config(base, self._load_yaml(path))

    def _merge_directory_overrides(self, base: Dict[str, Any], directory: Path) -> Dict[str, Any]:
        if not directory.exists():
            return base
        result = deepcopy(base)
        for path in sorted(directory.glob("*.yaml")):
            result = self._merge_yaml_if_exists(result, path)
        return result

    def _write_runtime_snapshot(self, config: Dict[str, Any], *, settings_path: Optional[Path] = None) -> None:
        runtime_dir = self.project_root / ".agentqms"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_config = runtime_dir / "effective.yaml"

        if settings_path is not None:
            layers: Dict[str, Any] = {
                "settings": str(settings_path.relative_to(self.project_root))
                if settings_path.is_relative_to(self.project_root)
                else str(settings_path)
            }
        else:
            layers = {
                "defaults": {
                    "framework": "AgentQMS/config_defaults/framework.yaml",
                    "interface": "AgentQMS/config_defaults/interface.yaml",
                    "paths": "AgentQMS/config_defaults/paths.yaml",
                    "tool_mappings": "AgentQMS/config_defaults/tool_mappings.json",
                },
                "project": {
                    "framework": "config/framework.yaml",
                    "interface": "config/interface.yaml",
                    "paths": "config/paths.yaml",
                    "environments": "config/environments/",
                    "overrides": "config/overrides/",
                },
            }

        payload = {
            "layers": layers,
            "metadata": {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "generator": "AgentQMS ConfigLoader",
                "schema_version": "0.2",
            },
            "resolved": config,
        }

        with runtime_config.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = deepcopy(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result


_config_loader: Optional[ConfigLoader] = None


def get_config_loader() -> ConfigLoader:
    """Return a singleton configuration loader."""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader


def load_config(force: bool = False) -> Dict[str, Any]:
    """Convenience helper for callers that only need the merged config."""
    return get_config_loader().load(force=force)
