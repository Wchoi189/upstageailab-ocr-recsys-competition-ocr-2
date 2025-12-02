"""Path resolution helpers for AgentQMS (agent_tools canonical).

All file-system lookups flow through this module to ensure consistent
path resolution across the framework.
"""

from __future__ import annotations

from pathlib import Path

from .config import get_config_loader, load_config


def get_framework_root() -> Path:
    """Return the root directory that contains the framework code."""
    return get_config_loader().framework_root


def get_project_root() -> Path:
    """Return the root directory of the host project."""
    return get_config_loader().project_root


def get_container_path(component_key: str) -> Path:
    """Return the path to a framework component defined in config."""
    config = load_config()
    relative = config.get("paths", {}).get(component_key)
    if not relative:
        raise KeyError(f"Unknown framework path key: {component_key}")

    return (get_framework_root() / relative).resolve()


def get_artifacts_dir() -> Path:
    """Return the configured artifacts directory."""
    return get_config_loader().get_path("artifacts")


def get_docs_dir() -> Path:
    """Return the configured documentation directory."""
    return get_config_loader().get_path("docs")


def get_agent_interface_dir() -> Path:
    """Return the path to the agent interface directory inside the framework."""
    return get_container_path("agent_interface")


def get_agent_tools_dir() -> Path:
    """Return the path to the implementation directory containing agent tools."""
    return get_container_path("implementation")


def get_project_conventions_dir() -> Path:
    """Return the path to the conventions directory inside the framework."""
    return get_container_path("project_conventions")


def get_project_config_dir() -> Path:
    """Return the root project config directory."""
    return get_project_root() / "config"


def ensure_within_project(path: Path) -> Path:
    """Ensure *path* resides within the project root (for safety checks)."""
    project_root = get_project_root().resolve()
    path = path.resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Path '{path}' escapes project root '{project_root}'") from exc
    return path


