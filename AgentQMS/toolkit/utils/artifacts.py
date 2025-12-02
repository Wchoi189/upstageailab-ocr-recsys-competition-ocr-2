"""Helpers for working with artifact categories and directories.

This module provides a small abstraction layer over the core path
configuration so callers can resolve well-known artifact categories
to concrete directories without hard-coding paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

from .config import load_config
from .paths import get_artifacts_dir


# Default mapping of artifact categories to subdirectories under the
# configured artifacts root. These values can be overridden in config
# via paths.artifact_categories.
DEFAULT_ARTIFACT_CATEGORIES: Dict[str, str] = {
    "implementation_plan": "implementation_plans",
    "assessment": "assessments",
    "design": "design_documents",
    "research": "research",
    "template": "templates",
    "bug_report": "bug_reports",
    "session_note": "completed_plans/completion_summaries/session_notes",
}


def get_artifacts_root() -> Path:
    """Return the configured artifacts root directory."""

    return get_artifacts_dir()


def get_artifact_categories() -> Mapping[str, str]:
    """Return the effective mapping of artifact categories to subdirs.

    Configuration can override or extend the defaults by defining
    a mapping under:

        paths:
          artifact_categories:
            implementation_plan: implementation_plans
            ...
    """

    config = load_config()
    configured = (
        config.get("paths", {}).get("artifact_categories") or {}
    )

    merged: Dict[str, str] = dict(DEFAULT_ARTIFACT_CATEGORIES)
    merged.update({str(k): str(v) for k, v in configured.items()})
    return merged


def get_artifact_category_dir(category: str) -> Path:
    """Return the directory for a specific artifact category.

    Raises KeyError if the category is unknown.
    """

    categories = get_artifact_categories()
    rel = categories.get(category)
    if not rel:
        raise KeyError(
            f"Unknown artifact category '{category}'. "
            f"Known categories: {', '.join(sorted(categories))}"
        )

    return (get_artifacts_root() / rel).resolve()


def get_known_artifact_categories() -> list[str]:
    """Return the list of known artifact category keys."""

    return sorted(get_artifact_categories().keys())


