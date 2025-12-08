"""Integration between artifact_workflow and tracking database.

This module handles auto-registration of artifacts in the tracking database
when they are created. It maps artifact types to tracking entity types
(plans, experiments, debug sessions, refactors) and registers them automatically.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from AgentQMS.agent_tools.utilities.tracking.db import (
    create_debug_session,
    upsert_experiment,
    upsert_feature_plan,
    upsert_refactor,
)

if TYPE_CHECKING:
    pass


# Mapping of artifact types to tracking entity types
ARTIFACT_TRACKING_MAPPING = {
    "implementation_plan": ("plan", upsert_feature_plan),
    "experiment": ("experiment", upsert_experiment),
    "debug_session": ("debug", create_debug_session),
    "audit": ("refactor", upsert_refactor),
}


def register_artifact_in_tracking(
    artifact_type: str,
    file_path: str | Path,
    title: str,
    owner: str | None = None,
    track_flag: bool = True,
) -> dict[str, Any]:
    """Register an artifact in the tracking database.

    Maps artifact metadata to tracking database entities based on artifact type.
    Returns a dict with tracking status and details.

    Args:
        artifact_type: Type of artifact (implementation_plan, experiment, etc.)
        file_path: Path to the artifact file
        title: Artifact title
        owner: Artifact owner/author (optional)
        track_flag: Whether to actually track this (default: True)

    Returns:
        dict with keys:
            - should_track: bool, whether this artifact type should be tracked
            - tracked: bool, whether registration succeeded
            - tracking_type: str, the entity type in tracking DB (plan, experiment, etc.)
            - tracking_key: str, the key generated for the entity
            - reason: str, reason for any failure
    """
    if not track_flag:
        return {
            "should_track": False,
            "tracked": False,
            "reason": "tracking disabled (track_flag=False)",
        }

    # Check if this artifact type is tracked
    if artifact_type not in ARTIFACT_TRACKING_MAPPING:
        return {
            "should_track": False,
            "tracked": False,
            "reason": f"artifact type '{artifact_type}' not in tracking system",
        }

    try:
        tracking_type, register_fn = ARTIFACT_TRACKING_MAPPING[artifact_type]

        # Generate a tracking key from the file path and title
        # Format: lowercase, hyphens instead of spaces/underscores, no special chars
        file_path_obj = Path(file_path)
        stem = file_path_obj.stem  # filename without extension
        tracking_key = f"{stem.lower().replace(' ', '-').replace('_', '-')}"

        # Register based on tracking type
        if tracking_type == "plan":
            entity_id = upsert_feature_plan(tracking_key, title, owner=owner)
        elif tracking_type == "experiment":
            entity_id = upsert_experiment(tracking_key, title, owner=owner)
        elif tracking_type == "debug":
            entity_id = create_debug_session(tracking_key, title)
        elif tracking_type == "refactor":
            entity_id = upsert_refactor(tracking_key, title)
        else:
            return {
                "should_track": True,
                "tracked": False,
                "tracking_type": tracking_type,
                "reason": f"unknown tracking type: {tracking_type}",
            }

        return {
            "should_track": True,
            "tracked": True,
            "tracking_type": tracking_type,
            "tracking_key": tracking_key,
            "entity_id": entity_id,
        }

    except Exception as e:
        return {
            "should_track": True,
            "tracked": False,
            "tracking_type": tracking_type,
            "reason": f"registration failed: {str(e)}",
        }


__all__ = ["register_artifact_in_tracking"]
