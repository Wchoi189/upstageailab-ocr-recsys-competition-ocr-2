#!/usr/bin/env python3
"""
Deep Validation Bridge

Restores strict validation by loading granular constraints from standards_db.json.
This bridges the gap between the new Markdown specs (light validation) and the
legacy YAML standards (strict validation).

Phase 7.3: Deep Validation Bridge
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class StrictConstraintValidator:
    """Validates artifacts using granular constraints from standards_db.json."""

    def __init__(self, project_root: Path):
        """Initialize with project root."""
        self.project_root = project_root
        self.constraints = self._load_strict_constraints()

    def _load_strict_constraints(self) -> dict[str, Any]:
        """Load granular constraints from standards_db.json."""
        db_path = self.project_root / "AgentQMS" / ".agentqms" / "standards_db.json"

        if not db_path.exists():
            return {}

        try:
            with db_path.open("r", encoding="utf-8") as f:
                db = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

        # Extract frontmatter schema from standards_db
        frontmatter_yaml = db.get("tier1-sst/specs/frontmatter-schema.yaml", "")
        if not frontmatter_yaml:
            return {}

        try:
            schema = yaml.safe_load(frontmatter_yaml)
            return schema.get("fields", {})
        except yaml.YAMLError:
            return {}

    def validate_field(self, field_name: str, value: Any) -> list[str]:
        """
        Validate a field value against strict constraints.

        Args:
            field_name: Name of the frontmatter field
            value: Value to validate

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if field_name not in self.constraints:
            return errors

        constraints = self.constraints[field_name]

        # Min words constraint
        if "min_words" in constraints and isinstance(value, str):
            words = len(value.split())
            min_words = constraints["min_words"]
            if words < min_words:
                errors.append(
                    f"{field_name}: too few words ({words} < {min_words})"
                )

        # Max words constraint
        if "max_words" in constraints and isinstance(value, str):
            words = len(value.split())
            max_words = constraints["max_words"]
            if words > max_words:
                errors.append(
                    f"{field_name}: too many words ({words} > {max_words})"
                )

        # Min length constraint
        if "min_length" in constraints and isinstance(value, str):
            min_len = constraints["min_length"]
            if len(value) < min_len:
                errors.append(
                    f"{field_name}: too short ({len(value)} < {min_len} chars)"
                )

        # Max length constraint
        if "max_length" in constraints and isinstance(value, str):
            max_len = constraints["max_length"]
            if len(value) > max_len:
                errors.append(
                    f"{field_name}: too long ({len(value)} > {max_len} chars)"
                )

        # Format validation (regex pattern)
        if "format" in constraints and isinstance(value, str):
            import re
            pattern = constraints["format"]
            if not re.match(pattern, value):
                errors.append(
                    f"{field_name}: does not match required format {pattern}"
                )

        return errors

    def validate_frontmatter(self, frontmatter: dict[str, Any]) -> list[str]:
        """
        Validate all frontmatter fields against strict constraints.

        Args:
            frontmatter: Frontmatter dictionary

        Returns:
            List of error messages (empty if valid)
        """
        all_errors = []

        for field_name, value in frontmatter.items():
            errors = self.validate_field(field_name, value)
            all_errors.extend(errors)

        return all_errors

    def is_available(self) -> bool:
        """Check if strict constraints are available."""
        return bool(self.constraints)
