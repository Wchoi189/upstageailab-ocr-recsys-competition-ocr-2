#!/usr/bin/env python3
"""
Tracking Database Integration for Artifact Workflow

Provides seamless integration between artifact creation/validation
and the tracking database for automatic registration and status sync.

Usage:
    from AgentQMS.agent_tools.utilities.tracking_integration import (
        register_artifact_in_tracking,
        sync_artifact_status,
        should_track_artifact,
    )
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

# Import from toolkit (legacy location) for now
try:
    from AgentQMS.toolkit.utilities.tracking.db import (
        get_connection,
        init_db,
        upsert_feature_plan,
    )
    from AgentQMS.toolkit.utilities.tracking.db import (
        set_plan_status as _set_plan_status,
    )

    TRACKING_DB_AVAILABLE = True
except ImportError:
    TRACKING_DB_AVAILABLE = False


def ensure_tracking_db_initialized() -> bool:
    """Ensure tracking database is initialized.

    Returns:
        True if tracking DB is available and initialized
    """
    if not TRACKING_DB_AVAILABLE:
        return False

    try:
        init_db()
        return True
    except Exception:
        return False


def should_track_artifact(artifact_type: str) -> bool:
    """Determine if artifact type should be auto-registered in tracking DB.

    Args:
        artifact_type: Artifact type (e.g., 'implementation_plan')

    Returns:
        True if should be tracked
    """
    # Track implementation plans, assessments, and bug reports by default
    trackable_types = {
        "implementation_plan",
        "assessment",
        "bug_report",
        "design",
        "research",
    }
    return artifact_type in trackable_types


def extract_artifact_key_from_path(file_path: Path) -> str | None:
    """Extract artifact key from file path.

    Args:
        file_path: Path to artifact file

    Returns:
        Extracted key or None
    """
    filename = file_path.name

    # Remove timestamp prefix (YYYY-MM-DD_HHMM_)
    pattern = r"^\d{4}-\d{2}-\d{2}_\d{4}_"
    cleaned = re.sub(pattern, "", filename)

    # Remove artifact type prefix
    type_prefixes = [
        "implementation_plan_",
        "assessment-",
        "audit-",
        "design-",
        "research-",
        "BUG_",
        "SESSION_",
    ]
    for prefix in type_prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]
            break

    # Remove extension
    cleaned = cleaned.rsplit(".", 1)[0]

    return cleaned if cleaned else None


def register_artifact_in_tracking(
    artifact_type: str,
    file_path: str | Path,
    title: str,
    owner: str | None = None,
    track_flag: bool = True,
) -> dict[str, Any]:
    """Register artifact in tracking database.

    Args:
        artifact_type: Type of artifact
        file_path: Path to created artifact
        title: Artifact title
        owner: Artifact owner
        track_flag: Whether to actually register (allows opt-out)

    Returns:
        Dictionary with registration result
    """
    result = {
        "tracked": False,
        "tracking_available": TRACKING_DB_AVAILABLE,
        "should_track": should_track_artifact(artifact_type),
        "track_flag": track_flag,
    }

    # Early returns for various reasons not to track
    if not track_flag:
        result["reason"] = "tracking disabled via flag"
        return result

    if not TRACKING_DB_AVAILABLE:
        result["reason"] = "tracking DB not available"
        return result

    if not should_track_artifact(artifact_type):
        result["reason"] = f"artifact type '{artifact_type}' not trackable"
        return result

    # Ensure DB is initialized
    if not ensure_tracking_db_initialized():
        result["reason"] = "failed to initialize tracking DB"
        return result

    # Extract artifact key
    path = Path(file_path)
    key = extract_artifact_key_from_path(path)

    if not key:
        result["reason"] = "could not extract artifact key from path"
        return result

    try:
        # Add artifact_path column if it doesn't exist
        _ensure_artifact_path_column()

        # Register based on artifact type
        if artifact_type == "implementation_plan":
            plan_id = upsert_feature_plan(key, title, owner)

            # Update artifact_path
            conn = get_connection()
            with conn:
                conn.execute(
                    "UPDATE feature_plans SET artifact_path = ? WHERE id = ?",
                    (str(path.absolute()), plan_id),
                )

            result.update(
                {
                    "tracked": True,
                    "tracking_type": "feature_plan",
                    "tracking_id": plan_id,
                    "tracking_key": key,
                    "artifact_path": str(path.absolute()),
                }
            )

        # TODO: Add support for other artifact types (experiments, debug_sessions, etc.)
        else:
            result["reason"] = f"tracking not yet implemented for {artifact_type}"

    except Exception as e:
        result["reason"] = f"tracking registration failed: {e}"

    return result


def _ensure_artifact_path_column() -> None:
    """Ensure artifact_path column exists in tracking tables."""
    conn = get_connection()
    with conn:
        # Check if column exists in feature_plans
        cursor = conn.execute("PRAGMA table_info(feature_plans)")
        columns = [row[1] for row in cursor.fetchall()]

        if "artifact_path" not in columns:
            # Add column (SQLite doesn't support adding with constraints in ALTER)
            conn.execute(
                "ALTER TABLE feature_plans ADD COLUMN artifact_path TEXT"
            )


def sync_artifact_status(
    file_path: str | Path, new_status: str
) -> dict[str, Any]:
    """Sync artifact status to tracking database.

    Args:
        file_path: Path to artifact
        new_status: New status value

    Returns:
        Sync result dictionary
    """
    result = {"synced": False, "tracking_available": TRACKING_DB_AVAILABLE}

    if not TRACKING_DB_AVAILABLE:
        result["reason"] = "tracking DB not available"
        return result

    path = Path(file_path)
    key = extract_artifact_key_from_path(path)

    if not key:
        result["reason"] = "could not extract artifact key"
        return result

    try:
        # Map artifact status to tracking status
        status_mapping = {
            "active": "in_progress",
            "draft": "pending",
            "completed": "completed",
            "archived": "completed",
            "deferred": "paused",
            "pending": "pending",
        }

        tracking_status = status_mapping.get(new_status, new_status)

        # Update status in tracking DB
        _set_plan_status(key, tracking_status)

        result.update(
            {
                "synced": True,
                "tracking_key": key,
                "artifact_status": new_status,
                "tracking_status": tracking_status,
            }
        )

    except Exception as e:
        result["reason"] = f"status sync failed: {e}"

    return result


def get_artifact_tracking_info(file_path: str | Path) -> dict[str, Any] | None:
    """Get tracking information for an artifact.

    Args:
        file_path: Path to artifact

    Returns:
        Tracking info dictionary or None if not tracked
    """
    if not TRACKING_DB_AVAILABLE:
        return None

    path = Path(file_path)
    key = extract_artifact_key_from_path(path)

    if not key:
        return None

    try:
        conn = get_connection(readonly=True)
        cursor = conn.execute(
            """
            SELECT id, key, title, status, owner, started_at, updated_at
            FROM feature_plans
            WHERE key = ?
            """,
            (key,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "key": row[1],
                "title": row[2],
                "status": row[3],
                "owner": row[4],
                "started_at": row[5],
                "updated_at": row[6],
            }

    except Exception:
        pass

    return None


def update_artifact_path_in_tracking(
    old_path: str | Path, new_path: str | Path
) -> dict[str, Any]:
    """Update artifact path in tracking database after move/rename.

    Args:
        old_path: Old artifact path
        new_path: New artifact path

    Returns:
        Update result dictionary
    """
    result = {"updated": False, "tracking_available": TRACKING_DB_AVAILABLE}

    if not TRACKING_DB_AVAILABLE:
        result["reason"] = "tracking DB not available"
        return result

    try:
        conn = get_connection()
        with conn:
            # Update using old path
            cursor = conn.execute(
                "UPDATE feature_plans SET artifact_path = ? WHERE artifact_path = ?",
                (str(Path(new_path).absolute()), str(Path(old_path).absolute())),
            )

            rows_updated = cursor.rowcount

            result.update(
                {
                    "updated": rows_updated > 0,
                    "rows_updated": rows_updated,
                    "old_path": str(old_path),
                    "new_path": str(new_path),
                }
            )

    except Exception as e:
        result["reason"] = f"path update failed: {e}"

    return result
