#!/usr/bin/env python3
"""
Artifact Validation Script for AI Agents

This script validates that artifacts follow the established naming conventions
and organizational structure defined in the project.

Supports extension via plugin system - see .agentqms/plugins/validators.yaml
Rules loaded from AgentQMS/knowledge/agent/artifact_rules.yaml

Usage:
    python validate_artifacts.py --check-naming
    python validate_artifacts.py --file path/to/artifact.md
    python validate_artifacts.py --directory artifacts/
    python validate_artifacts.py --all
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# Try to import context bundle functions for validation
try:
    from AgentQMS.agent_tools.core.context_bundle import (
        is_fresh,
        list_available_bundles,
        load_bundle_definition,
        validate_bundle_files,
    )

    CONTEXT_BUNDLES_AVAILABLE = True
except ImportError:
    CONTEXT_BUNDLES_AVAILABLE = False

# Try to import plugin registry for extensibility
try:
    from AgentQMS.agent_tools.core.plugins import get_plugin_registry

    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False

from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

from AgentQMS.agent_tools.compliance.validate_boundaries import BoundaryValidator  # noqa: E402
from AgentQMS.agent_tools.utils.paths import ensure_within_project, get_project_root


def load_artifact_rules() -> dict[str, Any] | None:
    """Load artifact rules from the YAML schema file."""
    try:
        rules_path = get_project_root() / "AgentQMS" / "knowledge" / "agent" / "artifact_rules.yaml"
        if rules_path.exists():
            with open(rules_path, encoding="utf-8") as f:
                return yaml.safe_load(f)
    except Exception:
        pass
    return None


# Load rules at module level (optional - fallback to builtins if not available)
ARTIFACT_RULES = load_artifact_rules()


def _assert_boundaries() -> None:
    violations = BoundaryValidator().validate()
    errors = [v for v in violations if v.severity == "error"]
    if errors:
        formatted = "\n".join(f"- {v.path}: {v.message}" for v in errors)
        raise RuntimeError(
            "Boundary validation failed. Resolve the following issues before running validators:\n"
            f"{formatted}"
        )


_assert_boundaries()


DATE_FORMAT = "%Y-%m-%d %H:%M (KST)"


class ArtifactValidator:
    """Validates artifacts against project naming conventions and structure.

    Supports extension via plugin system. Additional prefixes, types, categories,
    and statuses can be registered in .agentqms/plugins/validators.yaml
    """

    # Built-in defaults (always available)
    # Paths are relative to artifacts_root (docs/artifacts/)
    _BUILTIN_ARTIFACT_TYPES: dict[str, str] = {
        "implementation_plan_": "implementation_plans/",
        "assessment-": "assessments/",
        "audit-": "audits/",
        "design-": "design_documents/",
        "research-": "research/",
        "template-": "templates/",
        "BUG_": "bug_reports/",
        "SESSION_": "completed_plans/completion_summaries/session_notes/",
    }

    _BUILTIN_TYPES: list[str] = [
        "implementation_plan",
        "assessment",
        "design",
        "research",
        "template",
        "bug_report",
        "session_note",
        "completion_summary",
    ]

    _BUILTIN_CATEGORIES: list[str] = [
        "development",
        "architecture",
        "evaluation",
        "compliance",
        "code_quality",
        "reference",
        "planning",
        "research",
        "troubleshooting",
    ]

    _BUILTIN_STATUSES: list[str] = [
        "active",
        "draft",
        "completed",
        "archived",
        "deprecated",
    ]

    def __init__(self, artifacts_root: str | Path | None = None, strict_mode: bool = True):
        # Default to the configured artifacts directory if none is provided
        if artifacts_root is None:
            from AgentQMS.agent_tools.utils.paths import get_artifacts_dir

            self.artifacts_root = get_artifacts_dir().resolve()
        else:
            artifacts_root_path = Path(artifacts_root)
            if not artifacts_root_path.is_absolute():
                artifacts_root_path = get_project_root() / artifacts_root_path

            self.artifacts_root = ensure_within_project(artifacts_root_path.resolve())

        self.violations = []

        # Load rules from YAML schema if available
        self.rules = ARTIFACT_RULES
        self.rules_loaded = self.rules is not None

        # Strict mode controls validation strictness (default: True for strict validation)
        self.strict_mode = strict_mode

        # Load excluded directories from settings
        self.excluded_directories = self._load_excluded_directories()

        # Build artifact type mappings from rules or use builtins
        if self.rules and "artifact_types" in self.rules:
            self.valid_artifact_types = {}
            self.artifact_type_details = {}
            for type_name, type_def in self.rules["artifact_types"].items():
                prefix = type_def.get("prefix", "")
                directory = type_def.get("directory", "")
                if prefix and directory:
                    self.valid_artifact_types[prefix] = directory
                    self.artifact_type_details[prefix] = {
                        "name": type_name,
                        "separator": type_def.get("separator", "-"),
                        "case": type_def.get("case", "lowercase"),
                        "frontmatter_type": type_def.get("frontmatter_type", type_name),
                        "example": type_def.get("example", ""),
                        "description": type_def.get("description", ""),
                    }
        else:
            self.valid_artifact_types = dict(self._BUILTIN_ARTIFACT_TYPES)
            self.artifact_type_details = {}

        # Load frontmatter validation values from rules or use builtins
        if self.rules and "frontmatter" in self.rules:
            fm_rules = self.rules["frontmatter"]
            self.valid_statuses = fm_rules.get("valid_statuses", list(self._BUILTIN_STATUSES))
            self.valid_categories = fm_rules.get("valid_categories", list(self._BUILTIN_CATEGORIES))
            self.required_frontmatter = fm_rules.get("required_fields", ["title", "date", "type", "category", "status", "version"])
        else:
            self.valid_statuses = list(self._BUILTIN_STATUSES)
            self.valid_categories = list(self._BUILTIN_CATEGORIES)
            self.required_frontmatter = ["title", "date", "type", "category", "status", "version"]

        # Build valid types list from rules or use builtins
        if self.rules and "artifact_types" in self.rules:
            self.valid_types = [
                type_def.get("frontmatter_type", type_name)
                for type_name, type_def in self.rules["artifact_types"].items()
            ]
        else:
            self.valid_types = list(self._BUILTIN_TYPES)

        # Load error templates from rules
        self.error_templates = {}
        if self.rules and "error_templates" in self.rules:
            self.error_templates = self.rules["error_templates"]

        # Extend with plugin-registered values
        self._load_plugin_extensions()

    def _load_excluded_directories(self) -> list[str]:
        """Load excluded directories from settings.yaml."""
        try:
            from AgentQMS.agent_tools.utils.config import load_config

            config = load_config()
            return config.get("validation", {}).get("excluded_directories", ["archive", "deprecated"])
        except Exception:
            # Fallback to sensible defaults if config loading fails
            return ["archive", "deprecated"]

    def _is_excluded_path(self, file_path: Path) -> bool:
        """Check if a file path should be excluded from validation.

        Args:
            file_path: Path to check (can be absolute or relative)

        Returns:
            True if the path contains any excluded directory name
        """
        path_parts = file_path.parts
        return any(excluded_dir in path_parts for excluded_dir in self.excluded_directories)

    def _load_plugin_extensions(self) -> None:
        """Load additional validation rules from plugin registry."""
        if not PLUGINS_AVAILABLE:
            return

        try:
            registry = get_plugin_registry()
            validators = registry.get_validators()

            if not validators:
                return

            # Merge artifact types (plugin values override/extend builtin)
            if "prefixes" in validators:
                self.valid_artifact_types.update(validators["prefixes"])

            # Merge types (unique values)
            if "types" in validators:
                for t in validators["types"]:
                    if t not in self.valid_types:
                        self.valid_types.append(t)

            # Merge categories (unique values)
            if "categories" in validators:
                for c in validators["categories"]:
                    if c not in self.valid_categories:
                        self.valid_categories.append(c)

            # Merge statuses (unique values)
            if "statuses" in validators:
                for s in validators["statuses"]:
                    if s not in self.valid_statuses:
                        self.valid_statuses.append(s)

            # Also load prefixes from artifact type plugins
            artifact_types = registry.get_artifact_types()
            for name, type_def in artifact_types.items():
                validation = type_def.get("validation", {})
                prefix = validation.get("filename_prefix")
                directory = type_def.get("metadata", {}).get("directory")

                if prefix and directory:
                    self.valid_artifact_types[prefix] = directory

                # Add artifact type name to valid types
                if name not in self.valid_types:
                    self.valid_types.append(name)

        except Exception:
            # Plugin loading is non-critical - continue with builtins
            pass

    def _get_error_message(self, error_code: str, **kwargs) -> str:
        """Get formatted error message from templates or fallback."""
        # Find template by code
        template = None
        for name, tmpl in self.error_templates.items():
            if tmpl.get("code") == error_code:
                template = tmpl
                break

        if template:
            msg = template.get("message", "Validation error")
            expected = template.get("expected", "")
            hint = template.get("hint", "")

            # Format with kwargs
            try:
                msg = msg.format(**kwargs) if kwargs else msg
                expected = expected.format(**kwargs) if kwargs else expected
                hint = hint.format(**kwargs) if kwargs else hint
            except (KeyError, ValueError):
                pass  # Use unformatted if formatting fails

            parts = [f"[{error_code}] {msg}"]
            if expected:
                parts.append(f"Expected: {expected}")
            if hint:
                parts.append(f"Hint: {hint}")
            return " | ".join(parts)
        return f"[{error_code}] Validation error"

    def validate_timestamp_format(
        self, filename: str
    ) -> tuple[bool, str, re.Match | None]:
        """Validate timestamp format in filename."""
        timestamp_pattern = r"^\d{4}-\d{2}-\d{2}_\d{4}_"
        match = re.match(timestamp_pattern, filename)
        valid = bool(match)
        if valid:
            msg = "Valid timestamp format"
        else:
            msg = self._get_error_message("E001") if self.error_templates else \
                "Missing or invalid timestamp format (expected: YYYY-MM-DD_HHMM_)"
        match_result = match if valid else None
        return valid, msg, match_result

    def _detect_intended_type(self, after_timestamp: str) -> tuple[str, str] | None:
        """Try to detect what artifact type the user intended based on partial match."""
        after_lower = after_timestamp.lower()

        # Check for common misspellings or format errors
        type_hints = [
            ("assessment", "assessment-", "-"),
            ("implementation_plan", "implementation_plan_", "_"),
            ("implementation-plan", "implementation_plan_", "_"),  # Common mistake
            ("bug", "BUG_", "_"),
            ("bug_report", "BUG_", "_"),
            ("session", "SESSION_", "_"),
            ("design", "design-", "-"),
            ("research", "research-", "-"),
            ("audit", "audit-", "-"),
            ("template", "template-", "-"),
        ]

        for keyword, correct_prefix, expected_sep in type_hints:
            if after_lower.startswith(keyword):
                return (correct_prefix, expected_sep)

        return None

    def validate_naming_convention(self, file_path: Path) -> tuple[bool, str]:
        """Validate artifact naming convention."""
        filename = file_path.name

        # Check timestamp format
        valid, msg, match = self.validate_timestamp_format(filename)
        if not valid:
            return valid, msg

        after_timestamp = filename[match.end() :]

        # Check for valid artifact type after timestamp
        matched_type = None
        for artifact_type in self.valid_artifact_types:
            if after_timestamp.startswith(artifact_type):
                matched_type = artifact_type
                break

        if not matched_type:
            # Try to detect what type the user might have intended
            intended = self._detect_intended_type(after_timestamp)
            if intended:
                correct_prefix, expected_sep = intended
                type_details = self.artifact_type_details.get(correct_prefix, {})
                example = type_details.get("example", "")

                if self.error_templates:
                    msg = self._get_error_message(
                        "E003",
                        type=type_details.get("name", "artifact"),
                        expected_pattern=example,
                        separator=expected_sep,
                        case=type_details.get("case", "lowercase")
                    )
                else:
                    msg = f"Invalid format for artifact type. Use '{correct_prefix}' prefix. Example: {example}"
                return (False, msg)
            else:
                # No match found - list valid options
                valid_types_str = ", ".join(self.valid_artifact_types.keys())
                if self.error_templates:
                    msg = self._get_error_message("E002")
                else:
                    msg = f"Missing valid artifact type. Valid artifact types: {valid_types_str}"
                return (False, msg)

        # Check for kebab-case in descriptive part
        descriptive_part = after_timestamp[len(matched_type):-3]  # Remove .md
        type_details = self.artifact_type_details.get(matched_type, {})
        expected_case = type_details.get("case", "lowercase")

        # Check for ALL CAPS, uppercase, or mixed case words in descriptive part
        # Always validate - descriptive part must be lowercase for ALL types
        words = descriptive_part.replace("-", "_").split("_")
        has_uppercase_or_mixed = any(
            (word.isupper() or (not word.islower() and not word.isdigit())) and len(word) > 1
            for word in words
        )

        if descriptive_part.isupper() or has_uppercase_or_mixed:
            if expected_case == "uppercase_prefix":
                return (
                    False,
                    "Artifact filenames with uppercase prefix must have lowercase descriptive part. "
                    f"Example: {matched_type}001_lowercase-name.md (not {matched_type}001_UPPERCASE-NAME.md)",
                )
            else:
                return (
                    False,
                    "Artifact filenames must be lowercase. No ALL CAPS allowed. Use kebab-case (lowercase with hyphens)",
                )

        if (
            "_" in descriptive_part
            and not descriptive_part.replace("-", "").replace("_", "").isalnum()
        ):
            return (
                False,
                "Descriptive name should use kebab-case (hyphens, not underscores)",
            )

        return True, "Valid naming convention"

    def validate_directory_placement(self, file_path: Path) -> tuple[bool, str]:
        """Validate that file is in the correct directory."""
        filename = file_path.name
        relative_path = file_path.relative_to(self.artifacts_root)
        current_dir = str(relative_path.parent)

        # Validate timestamp format first
        timestamp_match = re.match(r'^\d{4}-\d{2}-\d{2}_\d{4}_', filename)
        if not timestamp_match:
            # File doesn't match timestamp-first format, can't validate directory
            return True, "Cannot validate directory (non-standard format)"

        # Extract everything after timestamp
        after_timestamp = filename[timestamp_match.end():]

        # Find which artifact type this file matches by checking if it starts with a registered type
        expected_dir = None
        matched_prefix = None
        for artifact_type, directory in self.valid_artifact_types.items():
            if after_timestamp.startswith(artifact_type):
                expected_dir = directory.rstrip("/")
                matched_prefix = artifact_type
                break

        if expected_dir and current_dir != expected_dir:
            type_details = self.artifact_type_details.get(matched_prefix, {})
            type_name = type_details.get("name", "artifact")

            if self.error_templates:
                msg = self._get_error_message("E004", expected_dir=expected_dir)
            else:
                msg = f"File should be in '{expected_dir}/' directory, currently in '{current_dir}/'"

            msg += f" (detected type: {type_name})"
            return (False, msg)

        return True, "Correct directory placement"

    def validate_artifacts_root(self, file_path: Path) -> tuple[bool, str]:
        """Ensure artifacts are in docs/artifacts/ not root /artifacts/."""
        try:
            # Get the path relative to project root
            from AgentQMS.agent_tools.utils.paths import get_project_root
            project_root = get_project_root()
            relative_path = file_path.relative_to(project_root)
            path_str = str(relative_path).replace("\\", "/")

            # Check if file starts with artifacts/ (without docs/ prefix)
            if path_str.startswith("artifacts/") and not path_str.startswith("docs/artifacts/"):
                return (
                    False,
                    f"Artifacts must be in 'docs/artifacts/' not root 'artifacts/'. "
                    f"Move file from '{path_str}' to 'docs/{path_str}'",
                )

            # Verify file is in docs/artifacts/ hierarchy
            if not path_str.startswith("docs/artifacts/") and not path_str.startswith("AgentQMS/"):
                return (
                    False,
                    f"Artifacts must be in 'docs/artifacts/' directory. "
                    f"Current location: '{path_str}'",
                )

        except ValueError:
            # File is outside project root - allow it (might be in AgentQMS module)
            pass

        return True, "Valid artifacts directory location"

    def validate_frontmatter(self, file_path: Path) -> tuple[bool, str]:
        """Validate frontmatter structure and content."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return False, f"Error reading file: {e}"

        # Check for frontmatter
        if not content.startswith("---"):
            return False, "Missing frontmatter (file should start with '---')"

        # Extract frontmatter
        frontmatter_end = content.find("---", 3)
        if frontmatter_end == -1:
            return False, "Malformed frontmatter (missing closing '---')"

        frontmatter_content = content[3:frontmatter_end]

        # Parse frontmatter (simple YAML-like parsing)
        frontmatter = {}
        for line in frontmatter_content.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                frontmatter[key] = value

        # Check required fields
        missing_fields = []
        for field in self.required_frontmatter:
            if field not in frontmatter:
                missing_fields.append(field)

        if missing_fields:
            return (
                False,
                f"Missing required frontmatter fields: {', '.join(missing_fields)}",
            )

        # Validate field values
        validation_errors = []

        date_value = frontmatter.get("date", "").strip()
        if date_value:
            try:
                datetime.strptime(date_value, DATE_FORMAT)
            except ValueError:
                validation_errors.append(
                    "Date must use 'YYYY-MM-DD HH:MM (KST)' format (24-hour clock)."
                )

        if "type" in frontmatter and frontmatter["type"] not in self.valid_types:
            validation_errors.append(
                f"Invalid type '{frontmatter['type']}'. Valid types: {', '.join(self.valid_types)}"
            )

        if (
            "category" in frontmatter
            and frontmatter["category"] not in self.valid_categories
        ):
            validation_errors.append(
                f"Invalid category '{frontmatter['category']}'. Valid categories: {', '.join(self.valid_categories)}"
            )

        if "status" in frontmatter and frontmatter["status"] not in self.valid_statuses:
            validation_errors.append(
                f"Invalid status '{frontmatter['status']}'. Valid statuses: {', '.join(self.valid_statuses)}"
            )

        if validation_errors:
            return False, "; ".join(validation_errors)

        return True, "Valid frontmatter"

    def _extract_frontmatter(self, file_path: Path) -> dict[str, str]:
        """Extract frontmatter from a file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            if not content.startswith("---"):
                return {}

            frontmatter_end = content.find("---", 3)
            if frontmatter_end == -1:
                return {}

            frontmatter_content = content[3:frontmatter_end]
            frontmatter = {}
            for line in frontmatter_content.split("\n"):
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip().strip("\"'")
                    frontmatter[key] = value
            return frontmatter
        except Exception:
            return {}

    def _get_type_from_filename(self, filename: str) -> str | None:
        """Extract artifact type from filename based on prefix."""
        timestamp_match = re.match(r'^\d{4}-\d{2}-\d{2}_\d{4}_', filename)
        if not timestamp_match:
            return None

        after_timestamp = filename[timestamp_match.end():]

        for prefix, _ in self.valid_artifact_types.items():
            if after_timestamp.startswith(prefix):
                type_details = self.artifact_type_details.get(prefix, {})
                return type_details.get("frontmatter_type", type_details.get("name", ""))

        return None

    def validate_type_consistency(self, file_path: Path) -> tuple[bool, str]:
        """Cross-validate frontmatter type against filename and directory.

        Phase 2: Ensures frontmatter `type:` matches the artifact type implied
        by the filename prefix and the expected directory.
        """
        filename = file_path.name

        # Extract frontmatter type
        frontmatter = self._extract_frontmatter(file_path)
        fm_type = frontmatter.get("type", "")

        # Extract type from filename
        filename_type = self._get_type_from_filename(filename)

        if not filename_type:
            # Can't determine type from filename, skip cross-validation
            return True, "Cannot determine type from filename"

        # Check if frontmatter type matches filename type
        if fm_type and fm_type != filename_type:
            # Build consolidated diagnostic
            mismatches = []
            mismatches.append(f"frontmatter type='{fm_type}'")
            mismatches.append(f"filename implies type='{filename_type}'")

            if self.error_templates:
                msg = self._get_error_message(
                    "E005",
                    fm_type=fm_type,
                    filename_type=filename_type,
                    expected_type=filename_type
                )
            else:
                msg = f"Type mismatch: {', '.join(mismatches)}. Update frontmatter type to '{filename_type}'"

            return False, msg

        return True, "Type consistency validated"

    def validate_single_file(self, file_path: Path, strict_mode: bool = None) -> dict:
        """Validate a single artifact file.

        Args:
            file_path: Path to the artifact file
            strict_mode: Override instance strict_mode setting. If None, uses self.strict_mode
        """
        # Use instance strict_mode if not explicitly overridden
        if strict_mode is None:
            strict_mode = self.strict_mode

        # Resolve relative paths to absolute
        if not file_path.is_absolute():
            file_path = file_path.resolve()

        result = {"file": str(file_path), "valid": True, "errors": []}

        # Skip INDEX.md files
        if file_path.name == "INDEX.md":
            return result

        # Validate artifacts root location (must be in docs/artifacts/)
        root_valid, root_msg = self.validate_artifacts_root(file_path)
        if not root_valid:
            result["valid"] = False
            result["errors"] += [f"Location: {root_msg}"]

        # Validate naming convention
        naming_valid, naming_msg = self.validate_naming_convention(file_path)
        if not naming_valid:
            if strict_mode:
                result["valid"] = False
                result["errors"] += [f"Naming: {naming_msg}"]
            else:
                result["errors"] += [f"Naming (lenient): {naming_msg}"]

        # Validate directory placement
        dir_valid, dir_msg = self.validate_directory_placement(file_path)
        if not dir_valid:
            if strict_mode:
                result["valid"] = False
                result["errors"] += [f"Directory: {dir_msg}"]
            else:
                result["errors"] += [f"Directory (lenient): {dir_msg}"]

        # Validate frontmatter
        frontmatter_valid, frontmatter_msg = self.validate_frontmatter(file_path)
        if not frontmatter_valid:
            if strict_mode:
                result["valid"] = False
                result["errors"] += [f"Frontmatter: {frontmatter_msg}"]
            else:
                result["errors"] += [f"Frontmatter (lenient): {frontmatter_msg}"]

        # Phase 2: Cross-validate frontmatter type with filename and directory
        type_valid, type_msg = self.validate_type_consistency(file_path)
        if not type_valid:
            if strict_mode:
                result["valid"] = False
                result["errors"] += [f"TypeConsistency: {type_msg}"]
            else:
                result["errors"] += [f"TypeConsistency (lenient): {type_msg}"]

        return result

    def validate_directory(self, directory: Path, strict_mode: bool = None) -> list[dict]:
        """Validate all markdown files in a directory.

        Args:
            directory: Directory path to validate
            strict_mode: Override instance strict_mode setting
        """
        results = []

        if not directory.exists():
            return [
                {
                    "file": str(directory),
                    "valid": False,
                    "errors": ["Directory does not exist"],
                }
            ]

        for file_path in directory.rglob("*.md"):
            if file_path.is_file():
                # Skip excluded directories (archive, deprecated, etc.)
                if self._is_excluded_path(file_path):
                    continue
                result = self.validate_single_file(file_path, strict_mode)
                results.append(result)

        return results

    def validate_all(self, strict_mode: bool = None) -> list[dict]:
        """Validate all artifacts in the artifacts directory.

        Args:
            strict_mode: Override instance strict_mode setting. If None, uses self.strict_mode
        """
        results = []

        # Validate all artifacts in subdirectories
        for subdirectory in self.artifacts_root.iterdir():
            if subdirectory.is_dir() and not subdirectory.name.startswith("_"):
                results.extend(self.validate_directory(subdirectory, strict_mode))

        # # Add bundle validation results if available
        # if CONTEXT_BUNDLES_AVAILABLE:
        #     bundle_results = self.validate_bundles()
        #     results.extend(bundle_results)

        return results

    def validate_bundles(self) -> list[dict]:
        """
        Validate context bundle definitions.

        Checks:
        - All bundle definition files exist and are valid YAML
        - All files referenced in bundles exist
        - All bundle files are fresh (modified within last 30 days)

        Returns:
            List of validation result dictionaries
        """
        if not CONTEXT_BUNDLES_AVAILABLE:
            return []

        results = []

        try:
            # Get project root using utility function
            from AgentQMS.agent_tools.utils.paths import get_project_root
            project_root = get_project_root()

            available_bundles = list_available_bundles()

            for bundle_name in available_bundles:
                # Determine bundle file path (framework or plugin)
                framework_bundle_path = project_root / "AgentQMS" / "knowledge" / "context_bundles" / f"{bundle_name}.yaml"
                plugin_bundle_path = project_root / ".agentqms" / "plugins" / "context_bundles" / f"{bundle_name}.yaml"

                if framework_bundle_path.exists():
                    bundle_file_display = f"AgentQMS/knowledge/context_bundles/{bundle_name}.yaml"
                elif plugin_bundle_path.exists():
                    bundle_file_display = f".agentqms/plugins/context_bundles/{bundle_name}.yaml"
                else:
                    bundle_file_display = f"context_bundles/{bundle_name}.yaml"

                bundle_result = {
                    "file": bundle_file_display,
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                }

                try:
                    # Load bundle definition
                    bundle_def = load_bundle_definition(bundle_name)

                    # Validate bundle files
                    validate_bundle_files(bundle_def)

                    # Check for missing files
                    tiers = bundle_def.get("tiers", {})

                    for tier_key, tier in tiers.items():
                        tier_files = tier.get("files", [])
                        for file_spec in tier_files:
                            if isinstance(file_spec, str):
                                file_path_str = file_spec
                                is_optional = False
                            elif isinstance(file_spec, dict):
                                file_path_str = file_spec.get("path", "")
                                is_optional = file_spec.get("optional", False)
                            else:
                                continue

                            # Skip glob patterns (handled by validate_bundle_files)
                            if "*" in file_path_str or "**" in file_path_str:
                                continue

                            # Check if file exists
                            file_path = project_root / file_path_str
                            if not file_path.exists():
                                if is_optional:
                                    # Optional files don't fail validation
                                    bundle_result["warnings"].append(
                                        f"Optional file missing in {bundle_name} bundle: {file_path_str}"
                                    )
                                else:
                                    bundle_result["valid"] = False
                                    bundle_result["errors"].append(
                                        f"Missing file in {bundle_name} bundle: {file_path_str}"
                                    )
                            elif not is_fresh(file_path, days=30):
                                bundle_result["warnings"].append(
                                    f"Stale file in {bundle_name} bundle: {file_path_str} "
                                    "(not modified in last 30 days)"
                                )

                except FileNotFoundError:
                    bundle_result["valid"] = False
                    bundle_result["errors"].append(
                        f"Bundle definition file not found: {bundle_name}.yaml"
                    )
                except Exception as e:
                    bundle_result["valid"] = False
                    bundle_result["errors"].append(
                        f"Error validating bundle {bundle_name}: {e!s}"
                    )

                results.append(bundle_result)

        except Exception as e:
            # Add error result if bundle system fails
            results.append(
                {
                    "file": "context_bundles/",
                    "valid": False,
                    "errors": [f"Error validating bundles: {e!s}"],
                    "warnings": [],
                }
            )

        return results

    def generate_report(self, results: list[dict]) -> str:
        """Generate a validation report with violation summary table."""
        total_files = len(results)
        valid_files = sum(1 for r in results if r["valid"])
        invalid_files = total_files - valid_files

        report = []
        report.append("=" * 60)
        report.append("ARTIFACT VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Total files: {total_files}")
        report.append(f"Valid files: {valid_files}")
        report.append(f"Invalid files: {invalid_files}")
        report.append(
            f"Compliance rate: {(valid_files / total_files * 100):.1f}%"
            if total_files > 0
            else "N/A"
        )
        report.append("")

        # Phase 6: Generate violation summary table
        if invalid_files > 0:
            # Count violations by type
            violation_counts: dict[str, int] = {}
            for result in results:
                if not result["valid"]:
                    for error in result.get("errors", []):
                        # Extract error code if present
                        code_match = re.search(r'\[E(\d+)\]', error)
                        if code_match:
                            code = f"E{code_match.group(1)}"
                        elif "Naming:" in error:
                            code = "Naming"
                        elif "Directory:" in error:
                            code = "Directory"
                        elif "Frontmatter:" in error:
                            code = "Frontmatter"
                        elif "TypeConsistency:" in error:
                            code = "TypeConsistency"
                        elif "Location:" in error:
                            code = "Location"
                        else:
                            code = "Other"

                        violation_counts[code] = violation_counts.get(code, 0) + 1

            report.append("VIOLATION SUMMARY:")
            report.append("-" * 40)
            report.append(f"{'Rule':<20} | {'Count':>6}")
            report.append("-" * 40)
            for code, count in sorted(violation_counts.items(), key=lambda x: -x[1]):
                report.append(f"{code:<20} | {count:>6}")
            report.append("-" * 40)
            report.append("")

            report.append("VIOLATIONS FOUND:")
            report.append("-" * 40)
            for result in results:
                if not result["valid"]:
                    report.append(f"\n❌ {result['file']}")
                    for error in result["errors"]:
                        report.append(f"   • {error}")

            # Add suggested next command
            report.append("")
            report.append("SUGGESTED NEXT COMMAND:")
            report.append("-" * 40)
            report.append(f"  cd AgentQMS/interface && make fix ARGS=\"--limit {min(invalid_files, 10)} --dry-run\"")
            report.append("")

        if valid_files > 0:
            report.append("\n✅ VALID FILES:")
            report.append("-" * 40)
            for result in results:
                if result["valid"]:
                    report.append(f"✓ {result['file']}")

        return "\n".join(report)

    def fix_suggestions(self, results: list[dict]) -> str:
        """Generate fix suggestions for invalid files."""
        suggestions = []
        suggestions.append("FIX SUGGESTIONS:")
        suggestions.append("=" * 40)

        for result in results:
            if not result["valid"]:
                suggestions.append(f"\n📁 {result['file']}")
                for error in result["errors"]:
                    if "Naming:" in error:
                        suggestions.append("   🔧 Rename file to follow convention:")
                        suggestions.append(
                            "      Format: YYYY-MM-DD_HHMM_{ARTIFACT_TYPE}_descriptive-name.md"
                        )
                    elif "Directory:" in error:
                        suggestions.append("   🔧 Move file to correct directory")
                    elif "Frontmatter:" in error:
                        suggestions.append("   🔧 Add or fix frontmatter:")
                        suggestions.append(
                            "      Required fields: title, date, type, category, status, version"
                        )

        return "\n".join(suggestions)


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate artifact naming conventions and structure"
    )
    parser.add_argument("--file", help="Validate a specific file")
    parser.add_argument("--directory", help="Validate all files in a directory")
    parser.add_argument("--all", action="store_true", help="Validate all artifacts")
    parser.add_argument(
        "--check-naming", action="store_true", help="Check naming conventions only"
    )
    parser.add_argument(
        "--artifacts-root",
        default=None,
        help="Root directory for artifacts",
    )
    parser.add_argument(
        "--staged",
        action="store_true",
        help="Validate only staged artifact files (for pre-commit/CI hooks)",
    )
    parser.add_argument("--output", help="Output file for report")
    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )
    parser.add_argument(
        "--lenient-plugins",
        action="store_true",
        help="Bypass strict validation mode for debugging (allows warnings without failures)",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to validate (positional arguments, for pre-commit hooks)",
    )

    args = parser.parse_args()

    # Determine strict mode (inverse of lenient)
    strict_mode = not args.lenient_plugins
    validator = ArtifactValidator(args.artifacts_root, strict_mode=strict_mode)
    artifacts_root = validator.artifacts_root

    if args.files:
        # Handle positional arguments (from pre-commit hooks passing explicit files)
        results = []
        for file_path_str in args.files:
            file_path = Path(file_path_str)
            if file_path.is_file():
                results.append(validator.validate_single_file(file_path))
            elif file_path.is_dir():
                results.extend(validator.validate_directory(file_path))
    elif args.staged:
        # Validate only staged files under the artifacts root (git required)
        import subprocess

        rel_artifacts_root = Path(artifacts_root)
        results = []

        try:
            completed = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                check=False,
                capture_output=True,
                text=True,
            )
            for line in (completed.stdout or "").splitlines():
                path = Path(line.strip())
                if not path.suffix.lower() == ".md":
                    continue
                # Only validate files within the artifacts tree
                try:
                    path.relative_to(rel_artifacts_root)
                except ValueError:
                    continue
                if path.exists():
                    results.append(validator.validate_single_file(path))
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"⚠️  Failed to determine staged files; falling back to --all: {exc}")
            results = validator.validate_all()
    elif args.file:
        file_path = Path(args.file)
        results = [validator.validate_single_file(file_path)]
    elif args.directory:
        dir_path = Path(args.directory)
        results = validator.validate_directory(dir_path)
    elif args.all:
        results = validator.validate_all()
    else:
        # Default: validate all
        results = validator.validate_all()

    if args.json:
        output = json.dumps(results, indent=2)
    else:
        output = validator.generate_report(results)
        if any(not r["valid"] for r in results):
            output += "\n\n" + validator.fix_suggestions(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    # Exit with error code if violations found
    if any(not r["valid"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
