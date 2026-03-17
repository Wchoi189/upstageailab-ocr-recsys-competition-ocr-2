"""Artifact versioning and aging helpers.

Provides the non-legacy import surface for artifact status reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re


@dataclass(frozen=True)
class SemanticVersion:
    major: int
    minor: int


class ArtifactAgeDetector:
    """Compute artifact age and lifecycle bucket."""

    def get_artifact_age(self, path: str | Path) -> int:
        target = Path(path)
        mtime = datetime.fromtimestamp(target.stat().st_mtime, tz=UTC)
        return max((datetime.now(UTC) - mtime).days, 0)

    def get_age_category(self, days: int) -> str:
        if days >= 365:
            return "archive"
        if days >= 180:
            return "stale"
        if days >= 90:
            return "warning"
        return "ok"


class VersionManager:
    """Extract semantic version from artifact frontmatter."""

    _VERSION_PATTERN = re.compile(r"^\s*version:\s*['\"]?(\d+)\.(\d+)['\"]?\s*$", re.IGNORECASE)

    def extract_version_from_frontmatter(self, path: str | Path) -> SemanticVersion | None:
        target = Path(path)
        if not target.exists():
            return None

        try:
            content = target.read_text(encoding="utf-8")
        except OSError:
            return None

        if not content.startswith("---"):
            return None

        for line in content.splitlines():
            if line.strip() == "---":
                continue
            match = self._VERSION_PATTERN.match(line)
            if match:
                return SemanticVersion(major=int(match.group(1)), minor=int(match.group(2)))
        return None
