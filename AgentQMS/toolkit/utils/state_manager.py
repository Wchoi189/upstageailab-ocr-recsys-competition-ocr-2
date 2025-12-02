"""Lightweight state manager for AgentQMS.

The v002_b release only needs a simple JSON-backed store, but the interface is
designed so we can swap in SQLite or an external database later without
changing downstream callers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from AgentQMS.toolkit.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

STATE_DIR = Path(".agentqms/state")
STATE_FILE = STATE_DIR / "agent_state.json"


class StateManager:
    """JSON-backed state manager with a future-proof interface."""

    def __init__(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Any] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def delete(self, key: str) -> None:
        self._state.pop(key, None)

    def save(self) -> None:
        STATE_FILE.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._state = data
            except json.JSONDecodeError:
                # Leave state empty; future versions could log telemetry
                self._state = {}


_DEFAULT_MANAGER: StateManager | None = None


def get_state_manager() -> StateManager:
    """Return a singleton state manager instance."""
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = StateManager()
    return _DEFAULT_MANAGER
