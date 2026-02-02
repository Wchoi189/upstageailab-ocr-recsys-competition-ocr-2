#!/usr/bin/env python3
"""
Artifact Validation Script for AI Agents

This script validates that artifacts follow the established naming conventions
and organizational structure defined in the project.

Supports extension via plugin system - see .agentqms/plugins/validators.yaml
Rules loaded from AgentQMS/knowledge/agent/artifact_rules.yaml

Usage:
    uv run validate_artifacts.py --check-naming
    uv run validate_artifacts.py --file path/to/artifact.md
    uv run validate_artifacts.py --directory artifacts/
    uv run validate_artifacts.py --all
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PreflightResult:
    """Result of pre-flight validation with structured error guidance."""

    valid: bool
    errors: list[dict[str, str]] = field(default_factory=list)

    def format_guidance(self) -> str:
        """Format errors as actionable guidance for agents."""
        if self.valid:
            return "✅ Pre-flight checks passed"

        lines = ["❌ Pre-flight validation failed:"]
        for e in self.errors:
            lines.append(f"  • {e.get('field', 'unknown')}: {e.get('error', 'Error')}")
            if e.get('fix'):
                lines.append(f"    → Fix: {e['fix']}")
            if e.get('example'):
                lines.append(f"    → Example: {e['example']}")
            if e.get('reference'):
                lines.append(f"    → See: {e['reference']}")
        return "\n".join(lines)

# Add project root to sys.path using runtime utility
from AgentQMS.tools.utils.system.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

# Try to import context bundle functions for validation


# Try to import plugin registry for extensibility
try:
    from AgentQMS.tools.core.plugins import get_plugin_registry

    PLUGINS_AVAILABLE = True
except ImportError:
    PLUGINS_AVAILABLE = False


def _refresh_plugin_snapshot_best_effort() -> None:
    """Refresh .agentqms/state/plugins.yaml if plugin system is available.

    This is intentionally best-effort and must never break validation runs.
    """

    try:
        from AgentQMS.tools.core.plugins import get_plugin_registry

        get_plugin_registry(force=False)
    except Exception:
        return


from AgentQMS.tools.compliance.validate_boundaries import BoundaryValidator  # noqa: E402
from AgentQMS.tools.utils.paths import ensure_within_project, get_project_root


# Artifact rules loading removed - validation now uses SpecParser (Phase 7.3)
# Legacy load_artifact_rules() function deleted - always returned None (path deleted in migration)
ARTIFACT_RULES = None  # Deprecated - kept for backward compatibility, always None


def _assert_boundaries() -> None:
    violations = BoundaryValidator().validate()
    errors = [v for v in violations if v.severity == "error"]
    if errors:
        formatted = "\n".join(f"- {v.path}: {v.message}" for v in errors)
        raise RuntimeError(f"Boundary validation failed. Resolve the following issues before running validators:\n{formatted}")


_assert_boundaries()



from AgentQMS.tools.compliance.validators.naming import validate_naming_convention
from AgentQMS.tools.compliance.validators.directory import validate_directory_placement, validate_artifacts_root
from AgentQMS.tools.compliance.validators.frontmatter import validate_frontmatter
from AgentQMS.tools.compliance.validators.consistency import validate_type_consistency

DATE_FORMAT = "%Y-%m-%d %H:%M (KST)"


class ArtifactValidator:
    """Validates artifacts against project naming conventions and structure.

    Supports extension via plugin system. Additional prefixes, types, categories,
    and statuses can be registered in .agentqms/plugins/validators.yaml
    """

    # Built-in defaults (always available)
    # Paths are relative to artifacts_root (docs/artifacts/)


    def __init__(self, artifacts_root: str | Path | None = None, strict_mode: bool = True):
        self._init_paths(artifacts_root)
        self.violations: list[dict[str, Any]] = []

        # Initialize Spec Parser (Assuming running from project root context)
        # We try to find project root:
        try:
            from AgentQMS.tools.utils.system.paths import get_project_root
            project_root = get_project_root()
        except ImportError:
            project_root = Path.cwd()

        from AgentQMS.tools.compliance.spec_parser import SpecParser
        self.spec_parser = SpecParser(project_root)

        # Phase 7.3: Initialize strict constraint validator (Deep Validation Bridge)
        from AgentQMS.tools.compliance.validators.strict_constraints import StrictConstraintValidator
        self.strict_validator = StrictConstraintValidator(project_root)

        self.strict_mode = strict_mode
        self.excluded_directories = self._load_excluded_directories()

        # Initialize rules dict before _init_type_mappings (fixes AttributeError)
        self.rules = {}

        self._init_type_mappings()
        self._init_frontmatter_rules()
        self._init_error_templates()
        self._load_plugin_extensions()

    def _init_paths(self, artifacts_root: str | Path | None) -> None:
        """Initialize artifact root paths."""
        if artifacts_root is None:
            from AgentQMS.tools.utils.paths import get_artifacts_dir
            self.artifacts_root = get_artifacts_dir().resolve()
        else:
            artifacts_root_path = Path(artifacts_root)
            if not artifacts_root_path.is_absolute():
                from AgentQMS.tools.utils.paths import get_project_root
                artifacts_root_path = get_project_root() / artifacts_root_path
            self.artifacts_root = ensure_within_project(artifacts_root_path.resolve())


    def _init_frontmatter_rules(self) -> None:
        """Initialize frontmatter validation rules from Spec-Kit."""
        self.valid_statuses = ["active", "draft", "completed", "archived", "deprecated"]
        self.valid_categories = ["development", "architecture", "evaluation", "compliance", "code_quality", "reference", "planning", "research", "troubleshooting"]

        # V2 Defaults (ADS 2.0) - Load from validation.spec.md
        parsed_fields = self.spec_parser.parse_required_fields()
        if parsed_fields:
            self.required_frontmatter = parsed_fields
        else:
            # Fallback if spec parsing fails
            self.required_frontmatter = ["title", "date", "type", "category", "status", "version", "ads_version"]

        # V1 Defaults (Legacy - no ads_version)
        self.legacy_required_frontmatter = [f for f in self.required_frontmatter if f != "ads_version"]

    def _init_type_mappings(self) -> None:
        """Initialize artifact type mappings from Spec-Kit."""
        # Load from compliance.spec.md table
        parsed_types = self.spec_parser.parse_artifact_types()

        if parsed_types:
            self.artifact_types = parsed_types
        else:
            # Fallback defaults if parsing fails
            self.artifact_types = {
                "implementation_plan": {"prefix": "implementation_plan_", "directory": "implementation_plans/"},
                "assessment": {"prefix": "assessment_", "directory": "assessments/"},
                "audit": {"prefix": "audit_", "directory": "audits/"},
                # ... add minimal fallbacks if needed
            }

        # Initialize valid_types list for validation checks (keys used as type)
        self.valid_types = list(self.artifact_types.keys())

        # Initialize artifact_type_details for naming validation
        # This is used by validate_naming_convention() function
        self.artifact_type_details = {}
        for type_name, type_info in self.artifact_types.items():
            prefix = type_info.get("prefix", "")
            if prefix:
                self.artifact_type_details[prefix] = {
                    "name": type_name,
                    "separator": type_info.get("separator", "-"),
                    "case": type_info.get("case", "lowercase"),
                    "frontmatter_type": type_info.get("frontmatter_type", type_name),
                    "example": type_info.get("example", ""),
                    "description": type_info.get("description", ""),
                }


    def _init_error_templates(self) -> None:
        """Initialize error templates."""
        # Default error templates
        self.error_templates = {
            "missing_timestamp": {
                "message": "Missing or invalid timestamp format",
                "expected": "Expected: YYYY-MM-DD_HHMM_ (e.g., 2025-11-29_1800_)",
                "hint": "Filename must start with date and time in format: YYYY-MM-DD_HHMM_"
            },
            "missing_type": {
                "message": "Missing artifact type prefix",
                "expected": "Expected recognized type prefix (e.g. implementation_plan_)",
                "hint": "Add the appropriate artifact type prefix after the timestamp"
            },
            "frontmatter_missing_field": {
                "message": "Missing required frontmatter field",
                "expected": "Required fields: title, date, type, category, status, version, ads_version",
                "hint": "Add missing field to frontmatter"
            },
            "frontmatter_type_mismatch": {
                "message": "Frontmatter type does not match filename",
                "hint": "Update frontmatter type to match filename prefix"
            }
        }

    def _load_excluded_directories(self) -> list[str]:
        """Load excluded directories from settings.yaml."""
        try:
            from AgentQMS.tools.utils.config import load_config

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

    def validate_preflight(
        self,
        artifact_type: str,
        name: str,
        title: str,
        frontmatter: dict[str, Any] | None = None,
    ) -> PreflightResult:
        """Validate artifact parameters BEFORE creation.

        Provides clear, actionable guidance for fixing any issues.

        Args:
            artifact_type: Type of artifact (e.g., 'assessment', 'implementation_plan')
            name: Descriptive name for the artifact (kebab-case)
            title: Human-readable title
            frontmatter: Optional additional frontmatter fields to validate

        Returns:
            PreflightResult with valid status and list of errors with fix guidance
        """
        errors: list[dict[str, str]] = []

        self._validate_artifact_type_param(artifact_type, errors)
        self._validate_name_param(name, errors)
        self._validate_title_param(title, errors)
        if frontmatter:
            self._validate_frontmatter_param(frontmatter, errors)

        return PreflightResult(valid=len(errors) == 0, errors=errors)

    def _validate_artifact_type_param(self, artifact_type: str, errors: list[dict[str, str]]) -> None:
        """Validate artifact_type parameter."""
        if artifact_type not in self.valid_types:
            similar = [t for t in self.valid_types if artifact_type[:3] in t][:3]
            errors.append({
                "field": "artifact_type",
                "error": f"Unknown type '{artifact_type}'",
                "fix": f"Use one of: {', '.join(self.valid_types[:6])}",
                "example": similar[0] if similar else "assessment",
                "reference": "AgentQMS/specs/tier1-contracts/compliance.spec.md",
            })

    def _validate_name_param(self, name: str, errors: list[dict[str, str]]) -> None:
        """Validate name parameter."""
        if not name:
            errors.append({
                "field": "name",
                "error": "Name is required",
                "fix": "Provide a descriptive name in kebab-case",
                "example": "api-improvements",
            })
            return

        if not re.match(r'^[a-z][a-z0-9-]*$', name):
            self._handle_invalid_name(name, errors)

    def _handle_invalid_name(self, name: str, errors: list[dict[str, str]]) -> None:
        """Handle breakdown of invalid name errors."""
        if name[0].isdigit():
            actual_issue = "starts with number"
        elif any(c.isupper() for c in name):
            actual_issue = "contains uppercase"
        else:
            actual_issue = "contains invalid characters"
        suggested_name = re.sub(r'[^a-z0-9-]', '-', name.lower()).strip('-')
        errors.append({
            "field": "name",
            "error": f"Name '{name}' {actual_issue}",
            "fix": "Use lowercase letters, numbers, and hyphens only. Start with a letter.",
            "example": suggested_name or "config-loader-improvements",
            "reference": "AgentQMS/specs/tier1-contracts/compliance.spec.md",
        })

    def _validate_title_param(self, title: str, errors: list[dict[str, str]]) -> None:
        """Validate title parameter."""
        if not title:
            errors.append({
                "field": "title",
                "error": "Title is required",
                "fix": "Provide a descriptive title (3-7 words recommended)",
                "example": "Implement Configuration Loading System",
            })
        elif len(title.split()) < 2:
            errors.append({
                "field": "title",
                "error": f"Title '{title}' too short (need 2+ words)",
                "fix": "Provide a more descriptive title (3-7 words)",
                "example": f"{title} Implementation" if title else "Feature Implementation Plan",
            })
        elif title.isupper():
            errors.append({
                "field": "title",
                "error": "Title should not be ALL CAPS",
                "fix": "Use Title Case or Sentence case",
                "example": title.title(),
            })

    def _validate_frontmatter_param(self, frontmatter: dict[str, Any], errors: list[dict[str, str]]) -> None:
        """Validate frontmatter parameter."""
        if "status" in frontmatter and frontmatter["status"] not in self.valid_statuses:
            errors.append({
                "field": "status",
                "error": f"Invalid status '{frontmatter['status']}'",
                "fix": f"Use one of: {', '.join(self.valid_statuses)}",
                "example": "draft",
            })
        if "category" in frontmatter and frontmatter["category"] not in self.valid_categories:
            errors.append({
                "field": "category",
                "error": f"Invalid category '{frontmatter['category']}'",
                "fix": f"Use one of: {', '.join(self.valid_categories)}",
                "example": "development",
            })

    def _load_plugin_extensions(self) -> None:
        """Load additional validation rules from plugin registry."""
        if not PLUGINS_AVAILABLE:
            return

        try:
            registry = get_plugin_registry()
            validators = registry.get_validators()

            if not validators:
                return

            self._merge_validator_config(validators)
            self._merge_artifact_type_plugins(registry)

        except Exception:
            # Plugin loading is non-critical - continue with builtins
            pass

    def _merge_validator_config(self, validators: dict[str, Any]) -> None:
        """Merge configuration from validators plugin."""
        # Merge artifact types (plugin values override/extend builtin)
        if "prefixes" in validators:
            self.artifact_types.update(validators["prefixes"])

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

    def _merge_artifact_type_plugins(self, registry: Any) -> None:
        """Merge definitions from artifact type plugins."""
        artifact_types = registry.get_artifact_types()
        for name, type_def in artifact_types.items():
            validation = type_def.get("validation", {})
            prefix = validation.get("filename_prefix")
            directory = type_def.get("metadata", {}).get("directory")

            if prefix and directory:
                self.artifact_types[prefix] = directory

            # Add artifact type name to valid types
            if name not in self.valid_types:
                self.valid_types.append(name)



    def validate_single_file(self, file_path: Path, strict_mode: bool | None = None) -> dict:
        """Validate a single artifact file.

        Args:
            file_path: Path to the artifact file
            strict_mode: Override instance strict_mode setting. If None, uses self.strict_mode
        """
        # Ensure path is absolute for comparison
        file_path = Path(file_path).resolve()

        # Mandatory scope check: ONLY validate files under the artifacts_root
        if not self._is_within_artifacts_root(file_path):
             return {
                "file": str(file_path),
                "valid": True,  # Technically not invalid if it's out of scope
                "skipped": True,
                "reason": f"File is outside configured artifacts root: {self.artifacts_root}",
            }

        # Use instance strict_mode if not explicitly overridden
        if strict_mode is None:
            strict_mode = self.strict_mode

        report_path = self._get_report_path(file_path)
        result: dict[str, Any] = {"file": report_path, "valid": True, "errors": []}

        # Skip INDEX.md files
        if file_path.name == "INDEX.md":
            return result

        if not self._validate_root_location(file_path, result):
            return result

        self._validate_naming(file_path, strict_mode, result)
        self._validate_directory_location(file_path, strict_mode, result)
        self._validate_file_frontmatter(file_path, strict_mode, result)
        self._validate_type_consistency(file_path, strict_mode, result)

        return result

    def _is_within_artifacts_root(self, file_path: Path) -> bool:
        try:
            file_path.relative_to(self.artifacts_root)
            return True
        except ValueError:
            return False

    def _get_report_path(self, file_path: Path) -> str:
        try:
            from AgentQMS.tools.utils.paths import get_project_root
            project_root = get_project_root()
            return str(file_path.relative_to(project_root))
        except (ValueError, ImportError):
            return str(file_path)

    def _validate_root_location(self, file_path: Path, result: dict[str, Any]) -> bool:
        root_valid, root_msg = validate_artifacts_root(file_path)
        if not root_valid:
            result["valid"] = False
            result["errors"] += [f"Location: {root_msg}"]
            return False
        return True

    def _validate_naming(self, file_path: Path, strict_mode: bool, result: dict[str, Any]) -> None:
        naming_valid, naming_msg = validate_naming_convention(
            file_path,
            self.artifact_types,
            self.artifact_type_details,
            self.error_templates
        )
        if not naming_valid:
            prefix = "Naming" if strict_mode else "Naming (lenient)"
            if strict_mode:
                result["valid"] = False
            result["errors"] += [f"{prefix}: {naming_msg}"]

    def _validate_directory_location(self, file_path: Path, strict_mode: bool, result: dict[str, Any]) -> None:
        dir_valid, dir_msg = validate_directory_placement(
            file_path,
            self.artifacts_root,
            self.artifact_types,
            self.artifact_type_details,
            self.error_templates
        )
        if not dir_valid:
            prefix = "Directory" if strict_mode else "Directory (lenient)"
            if strict_mode:
                result["valid"] = False
            result["errors"] += [f"{prefix}: {dir_msg}"]

    def _validate_file_frontmatter(self, file_path: Path, strict_mode: bool, result: dict[str, Any]) -> None:
        # Determine strictness based on version presence
        required_fields = self.required_frontmatter # Default to V2

        try:
            # Peek for ads_version
            has_ads_version = False
            with open(file_path, encoding="utf-8") as f:
                line = f.readline().strip()
                if line == "---":
                    for line in f:
                        line = line.strip()
                        if line == "---":
                            break
                        if line.startswith("ads_version:") or line.startswith("'ads_version':") or line.startswith('"ads_version":'):
                            has_ads_version = True
                            break

            if not has_ads_version:
                required_fields = self.legacy_required_frontmatter

        except Exception:
            # Fallback to default (V2) on read error
            pass

        frontmatter_valid, frontmatter_msg = validate_frontmatter(
            file_path,
            self.valid_statuses,
            self.valid_categories,
            self.valid_types,
            required_fields
        )
        if not frontmatter_valid:
            prefix = "Frontmatter" if strict_mode else "Frontmatter (lenient)"
            if strict_mode:
                result["valid"] = False
            result["errors"] += [f"{prefix}: {frontmatter_msg}"]

        # Phase 7.3: Deep Validation Bridge - Apply strict constraints if available
        if strict_mode and self.strict_validator.is_available():
            try:
                import yaml
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                    # Extract frontmatter
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        if len(parts) >= 3:
                            frontmatter = yaml.safe_load(parts[1])
                            if frontmatter:
                                constraint_errors = self.strict_validator.validate_frontmatter(frontmatter)
                                if constraint_errors:
                                    result["valid"] = False
                                    for err in constraint_errors:
                                        result["errors"] += [f"StrictConstraint: {err}"]
            except Exception:
                # Graceful degradation - continue with spec-only validation
                pass

    def _validate_type_consistency(self, file_path: Path, strict_mode: bool, result: dict[str, Any]) -> None:
        type_valid, type_msg = validate_type_consistency(
            file_path,
            self.artifact_types,
            self.artifact_type_details,
            self.error_templates
        )
        if not type_valid:
            prefix = "TypeConsistency" if strict_mode else "TypeConsistency (lenient)"
            if strict_mode:
                result["valid"] = False
            result["errors"] += [f"{prefix}: {type_msg}"]

    def validate_directory(self, directory: Path, strict_mode: bool | None = None) -> list[dict]:
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

    # Combined validate_all that correctly discovers all files recursively from root
    def validate_all(self, strict_mode: bool | None = None) -> list[dict]:
        """Validate all artifacts in the artifacts directory recursively.

        Args:
            strict_mode: Override instance strict_mode setting. If None, uses self.strict_mode
        """
        return self.validate_directory(self.artifacts_root, strict_mode)

    def check_naming_conventions(self) -> bool:
        """Check naming conventions for all artifacts (CLI support)."""
        all_valid = True
        violations = []

        # Iterate all files similar to validate_all
        for subdirectory in self.artifacts_root.iterdir():
            if subdirectory.is_dir() and not subdirectory.name.startswith("_"):
                subdir_violations = self._check_subdir_naming(subdirectory)
                if subdir_violations:
                    all_valid = False
                    violations.extend(subdir_violations)

        if violations:
            print("\nNaming Convention Violations:")
            for v in violations:
                print(v)
            return False

        print("✅ All artifacts follow naming conventions.")
        return True

    def _check_subdir_naming(self, subdirectory: Path) -> list[str]:
        """Check naming conventions for files in a subdirectory."""
        from AgentQMS.tools.compliance.validators.naming import validate_naming_convention
        violations = []
        for file_path in subdirectory.rglob("*.md"):
            if file_path.is_file() and not self._is_excluded_path(file_path):
                naming_valid, naming_msg = validate_naming_convention(
                    file_path,
                    self.artifact_types,
                    self.artifact_type_details,
                    self.error_templates
                )
                if not naming_valid:
                    rel_path = str(file_path.relative_to(self.artifacts_root))
                    violations.append(f"❌ {rel_path}: {naming_msg}")
        return violations







def _parse_cli_args():
    parser = argparse.ArgumentParser(description="Validate artifact naming conventions and structure")
    parser.add_argument("--file", help="Validate a specific file")
    parser.add_argument("--directory", help="Validate all files in a directory")
    parser.add_argument("--all", action="store_true", help="Validate all artifacts")
    parser.add_argument("--check-naming", action="store_true", help="Check naming conventions only")
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
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
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
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip plugin snapshot refresh (useful for pre-commit hooks)",
    )
    return parser.parse_args()


def _get_staged_files(artifacts_root: Path) -> list[Path]:
    """Discover staged markdown files under the artifacts root."""
    import subprocess

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
            if path.suffix.lower() != ".md":
                continue
            # Only validate files within the artifacts tree
            try:
                path.relative_to(artifacts_root)
            except ValueError:
                continue
            if path.exists():
                results.append(path)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"⚠️  Failed to determine staged files; falling back to --all: {exc}")
        return []

    return results


def _run_validations(validator: ArtifactValidator, args: argparse.Namespace) -> list[dict]:
    """Dispatch validations based on CLI arguments."""
    results = []
    if args.files:
        for file_path_str in args.files:
            file_path = Path(file_path_str)
            if file_path.is_file():
                results.append(validator.validate_single_file(file_path))
            elif file_path.is_dir():
                results.extend(validator.validate_directory(file_path))
    elif args.staged:
        staged_files = _get_staged_files(validator.artifacts_root)
        if not staged_files:
            results = validator.validate_all()
        else:
            for path in staged_files:
                results.append(validator.validate_single_file(path))
    elif args.file:
        results = [validator.validate_single_file(Path(args.file))]
    elif args.directory:
        results = validator.validate_directory(Path(args.directory))
    else:
        # Default, --all or no args
        results = validator.validate_all()
    return results


def _process_results(results: list[dict], args: argparse.Namespace) -> None:
    """Format and output validation results."""
    import json
    if not results:
        artifacts_root = getattr(args, 'artifacts_root', None) or "default artifacts directory"
        print(f"⚠️  No markdown files found for validation in {artifacts_root}")
        print("💡 Ensure you are pointing to the correct --artifacts-root or that your files have .md extension.")
        return

    if args.json:
        output = json.dumps(results, indent=2)
    else:
        from AgentQMS.tools.compliance.reporting import fix_suggestions, generate_report
        output = generate_report(results)
        if any(not r["valid"] for r in results):
            output += "\n\n" + fix_suggestions(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)

    # Flush stdout to ensure redirection captures all output
    import sys
    sys.stdout.flush()

    # Exit with error code if violations found
    if any(not r["valid"] for r in results):
        sys.exit(1)


def main():
    """Main entry point for the validation script."""
    args = _parse_cli_args()

    if not args.no_refresh:
        _refresh_plugin_snapshot_best_effort()

    # Determine strict mode (inverse of lenient)
    strict_mode = not args.lenient_plugins
    validator = ArtifactValidator(args.artifacts_root, strict_mode=strict_mode)

    results = _run_validations(validator, args)
    _process_results(results, args)


if __name__ == "__main__":
    main()
