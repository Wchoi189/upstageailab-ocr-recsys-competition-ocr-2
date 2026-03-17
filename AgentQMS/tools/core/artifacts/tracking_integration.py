"""Artifact tracking compatibility module.

Defines the stable tracking boundary used by artifact workflow code after the
legacy utilities package removal.
"""

from __future__ import annotations

from typing import Any


def register_artifact_in_tracking(
    artifact_type: str,
    file_path: str,
    title: str,
    owner: str | None,
    *,
    track_flag: bool = True,
) -> dict[str, Any]:
    """Best-effort tracking hook.

    Current implementation is intentionally no-op to avoid hard dependency on
    removed legacy integrations while preserving caller contracts.
    """
    _ = (artifact_type, file_path, title, owner)
    return {
        "tracked": False,
        "should_track": bool(track_flag),
        "reason": "tracking integration not configured",
        "tracking_type": None,
        "tracking_key": None,
    }
