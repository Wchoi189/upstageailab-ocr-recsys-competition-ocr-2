"""Migration + compatibility helpers for the framework restructure."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Set
import sys

MIGRATION_LOG = Path(".agentqms") / "migration.log"
_EMITTED_MESSAGES: Set[str] = set()


def log_migration_event(message: str) -> None:
    """Write a migration event to stderr and the migration log."""
    if message in _EMITTED_MESSAGES:
        return
    _EMITTED_MESSAGES.add(message)

    timestamp = datetime.now(timezone.utc).isoformat()
    payload = f"[{timestamp}] {message}"
    print(f"[AgentQMS migration] {message}", file=sys.stderr)
    MIGRATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with MIGRATION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(payload + "\n")


def warn_if_legacy_directories(framework_root: Path) -> None:
    """Log a warning when deprecated directories are still present."""
    replacements: Iterable[Tuple[str, str]] = (
        ("agent", "agent_interface"),
        ("conventions", "project_conventions"),
    )
    for old_name, new_name in replacements:
        old_path = framework_root / old_name
        if old_path.exists():
            new_path = framework_root / new_name
            log_migration_event(
                f"Legacy directory '{old_path}' detected. "
                f"Please migrate to '{new_path}' and remove the old path."
            )


def warn_if_legacy_paths(config: Dict[str, Any]) -> None:
    """Log a warning when configuration points to deprecated directories."""
    paths = config.get("paths", {}) or {}
    mappings: Iterable[Tuple[str, str, str]] = (
        ("agent_interface", "agent", "AgentQMS/agent_interface"),
        ("project_conventions", "conventions", "AgentQMS/project_conventions"),
    )
    for key, legacy_value, recommended in mappings:
        if paths.get(key) == legacy_value:
            log_migration_event(
                f"Path '{key}' still points to legacy value '{legacy_value}'. "
                f"Update project config to use '{recommended}'."
            )

