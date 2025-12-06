#!/usr/bin/env python3
"""
Versioning and Lifecycle Management for AgentQMS Artifacts

Handles semantic versioning (MAJOR.MINOR), lifecycle transitions,
and artifact aging detection.

Usage:
    from AgentQMS.agent_tools.utilities.versioning import (
        SemanticVersion, ArtifactLifecycle, ArtifactAgeDetector
    )
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml


@dataclass
class SemanticVersion:
    """Semantic version representation (MAJOR.MINOR)."""

    major: int
    minor: int

    @classmethod
    def from_string(cls, version_str: str) -> "SemanticVersion":
        """Parse version string like '1.0' or '2.3'."""
        if not version_str:
            return cls(1, 0)  # Default to 1.0

        # Handle "1.0", "1", or similar formats
        parts = str(version_str).strip().split(".")

        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            return cls(major, minor)
        except (ValueError, IndexError):
            # Invalid format, default to 1.0
            return cls(1, 0)

    def __str__(self) -> str:
        """Return string representation (e.g., '1.0')."""
        return f"{self.major}.{self.minor}"

    def bump_major(self) -> "SemanticVersion":
        """Increment major version, reset minor to 0."""
        return SemanticVersion(self.major + 1, 0)

    def bump_minor(self) -> "SemanticVersion":
        """Increment minor version."""
        return SemanticVersion(self.major, self.minor + 1)

    def is_newer_than(self, other: "SemanticVersion") -> bool:
        """Check if this version is newer than another."""
        if self.major != other.major:
            return self.major > other.major
        return self.minor > other.minor


class ArtifactLifecycle:
    """Manages artifact lifecycle state transitions."""

    # Valid lifecycle states
    VALID_STATUSES = ["draft", "active", "superseded", "archived"]

    # Valid transitions: from_status -> [allowed_to_statuses]
    VALID_TRANSITIONS = {
        "draft": ["active", "archived"],
        "active": ["superseded", "archived"],
        "superseded": ["archived"],
        "archived": [],  # Terminal state
    }

    @staticmethod
    def is_valid_status(status: str) -> bool:
        """Check if status is valid."""
        return status.lower() in ArtifactLifecycle.VALID_STATUSES

    @staticmethod
    def can_transition(from_status: str, to_status: str) -> bool:
        """Check if transition is allowed."""
        from_status = from_status.lower()
        to_status = to_status.lower()

        if from_status not in ArtifactLifecycle.VALID_TRANSITIONS:
            return False

        return to_status in ArtifactLifecycle.VALID_TRANSITIONS[from_status]

    @staticmethod
    def get_allowed_transitions(from_status: str) -> list[str]:
        """Get list of valid transition targets."""
        from_status = from_status.lower()
        return ArtifactLifecycle.VALID_TRANSITIONS.get(from_status, [])


class ArtifactAgeDetector:
    """Detects and categorizes artifact aging."""

    # Aging thresholds (in days)
    THRESHOLD_WARNING = 90  # Yellow: 90+ days without update
    THRESHOLD_STALE = 180  # Orange: 180+ days without update
    THRESHOLD_ARCHIVE = 365  # Red: 365+ days without update

    @staticmethod
    def get_artifact_age(artifact_path: Path) -> int | None:
        """Get age of artifact in days from last modification.

        Args:
            artifact_path: Path to artifact file

        Returns:
            Age in days, or None if file not found
        """
        if not artifact_path.exists():
            return None

        file_mtime = artifact_path.stat().st_mtime
        file_date = datetime.fromtimestamp(file_mtime)
        age = (datetime.now() - file_date).days

        return age

    @staticmethod
    def get_age_category(age_days: int) -> str:
        """Categorize age level.

        Args:
            age_days: Age in days

        Returns:
            Category: "fresh", "warning", "stale", or "archive"
        """
        if age_days < ArtifactAgeDetector.THRESHOLD_WARNING:
            return "fresh"
        elif age_days < ArtifactAgeDetector.THRESHOLD_STALE:
            return "warning"
        elif age_days < ArtifactAgeDetector.THRESHOLD_ARCHIVE:
            return "stale"
        else:
            return "archive"

    @staticmethod
    def get_age_emoji(age_days: int) -> str:
        """Get emoji representing age category.

        Args:
            age_days: Age in days

        Returns:
            Emoji: 🟢 fresh, 🟡 warning, 🟠 stale, 🔴 archive
        """
        category = ArtifactAgeDetector.get_age_category(age_days)
        emojis = {"fresh": "🟢", "warning": "🟡", "stale": "🟠", "archive": "🔴"}
        return emojis.get(category, "❓")

    @staticmethod
    def analyze_artifact(artifact_path: Path) -> dict:
        """Analyze artifact aging and return report.

        Args:
            artifact_path: Path to artifact file

        Returns:
            Dictionary with age info
        """
        age = ArtifactAgeDetector.get_artifact_age(artifact_path)

        if age is None:
            return {"valid": False, "error": "File not found"}

        category = ArtifactAgeDetector.get_age_category(age)
        emoji = ArtifactAgeDetector.get_age_emoji(age)

        return {
            "valid": True,
            "age_days": age,
            "category": category,
            "emoji": emoji,
            "last_modified": datetime.fromtimestamp(artifact_path.stat().st_mtime).isoformat(),
            "recommendation": ArtifactAgeDetector.get_recommendation(category),
        }

    @staticmethod
    def get_recommendation(category: str) -> str:
        """Get recommendation based on age category.

        Args:
            category: Age category

        Returns:
            Recommendation text
        """
        recommendations = {
            "fresh": "No action needed. Keep active.",
            "warning": "Consider reviewing and updating content.",
            "stale": "Should be reviewed, updated, or archived soon.",
            "archive": "Recommend archiving if no longer relevant.",
        }
        return recommendations.get(category, "Unknown category")


class VersionManager:
    """Manages version changes and history."""

    @staticmethod
    def extract_version_from_frontmatter(artifact_path: Path) -> SemanticVersion | None:
        """Extract version from artifact frontmatter.

        Args:
            artifact_path: Path to artifact file

        Returns:
            SemanticVersion or None if not found
        """
        try:
            with open(artifact_path) as f:
                content = f.read()

            # Extract frontmatter (between --- markers)
            if not content.startswith("---"):
                return None

            end_marker = content.find("---", 3)
            if end_marker == -1:
                return None

            frontmatter_str = content[3:end_marker]
            frontmatter = yaml.safe_load(frontmatter_str)

            if not frontmatter or "version" not in frontmatter:
                return None

            return SemanticVersion.from_string(frontmatter["version"])

        except Exception:
            return None

    @staticmethod
    def update_version_in_frontmatter(
        artifact_path: Path, new_version: SemanticVersion
    ) -> bool:
        """Update version in artifact frontmatter.

        Args:
            artifact_path: Path to artifact file
            new_version: New semantic version

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(artifact_path) as f:
                content = f.read()

            if not content.startswith("---"):
                return False

            end_marker = content.find("---", 3)
            if end_marker == -1:
                return False

            frontmatter_str = content[3:end_marker]
            body = content[end_marker + 3 :]

            frontmatter = yaml.safe_load(frontmatter_str)
            frontmatter["version"] = str(new_version)

            # Reconstruct file with updated frontmatter
            updated_content = "---\n" + yaml.dump(frontmatter) + "---\n" + body

            with open(artifact_path, "w") as f:
                f.write(updated_content)

            return True

        except Exception:
            return False


class VersionValidator:
    """Validates version formats and consistency."""

    VERSION_PATTERN = re.compile(r"^\d+\.\d+$")

    @staticmethod
    def is_valid_version_format(version_str: str) -> bool:
        """Check if version matches MAJOR.MINOR format.

        Args:
            version_str: Version string to validate

        Returns:
            True if valid format
        """
        return bool(VersionValidator.VERSION_PATTERN.match(str(version_str).strip()))

    @staticmethod
    def validate_version_consistency(
        artifact_path: Path, artifact_type: str
    ) -> tuple[bool, str]:
        """Validate version consistency for artifact.

        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact (e.g., 'assessment', 'implementation_plan')

        Returns:
            Tuple of (is_valid, message)
        """
        version = VersionManager.extract_version_from_frontmatter(artifact_path)

        if not version:
            return False, "Missing version in frontmatter"

        if not VersionValidator.is_valid_version_format(str(version)):
            return False, f"Invalid version format: {version} (expected MAJOR.MINOR)"

        # For new artifacts, version should be 1.0
        # For updated artifacts, version should be > 1.0
        if version.major < 1:
            return False, "Major version must be >= 1"

        return True, "Version valid"
