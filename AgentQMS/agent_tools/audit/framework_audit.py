#!/usr/bin/env python3
"""
Framework Audit Tool for AgentQMS

Scans plugin and template definitions for schema compliance.
Validates that all plugins conform to required schemas and naming conventions.

Usage:
    python framework_audit.py
    python framework_audit.py --verbose
    python framework_audit.py --json
    python framework_audit.py --plugins-only
    python framework_audit.py --templates-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


class FrameworkAudit:
    """Audit framework plugins and templates for schema compliance."""

    REQUIRED_PLUGIN_FIELDS = {
        "artifact_types": ["name", "version", "description", "scope", "metadata", "validation"],
        "validators": ["version", "description"],
        "context_bundles": ["version", "description"],
    }

    REQUIRED_METADATA_FIELDS = {
        "artifact_types": ["filename_pattern", "directory", "frontmatter"],
    }

    REQUIRED_VALIDATION_FIELDS = {
        "artifact_types": ["required_fields", "filename_prefix"],
    }

    def __init__(self, verbose: bool = False):
        """Initialize framework audit.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.project_root = self._get_project_root()
        self.plugins_root = self.project_root / ".agentqms" / "plugins"
        self.artifacts_root = self.project_root / "docs" / "artifacts"
        self.results = {
            "plugin_artifacts": [],
            "plugin_validators": [],
            "plugin_bundles": [],
            "templates": [],
            "summary": {
                "total_checked": 0,
                "invalid_count": 0,
                "warning_count": 0,
            }
        }

    def _get_project_root(self) -> Path:
        """Get project root directory."""
        from AgentQMS.agent_tools.utils.paths import get_project_root
        return get_project_root()

    def audit_plugins(self) -> list[dict[str, Any]]:
        """Audit all plugin definitions for schema compliance."""
        issues = []

        # Audit artifact type plugins
        artifact_types_dir = self.plugins_root / "artifact_types"
        if artifact_types_dir.exists():
            for plugin_file in artifact_types_dir.glob("*.yaml"):
                issue = self._audit_artifact_type_plugin(plugin_file)
                if issue:
                    issues.append(issue)
                self.results["plugin_artifacts"].append(issue or {"file": str(plugin_file), "valid": True})

        # Audit validators plugin
        validators_file = self.plugins_root / "validators.yaml"
        if validators_file.exists():
            issue = self._audit_validators_plugin(validators_file)
            if issue:
                issues.append(issue)
            self.results["plugin_validators"].append(issue or {"file": str(validators_file), "valid": True})

        # Audit context bundles
        bundles_dir = self.plugins_root / "context_bundles"
        if bundles_dir.exists():
            for bundle_file in bundles_dir.glob("*.yaml"):
                issue = self._audit_context_bundle(bundle_file)
                if issue:
                    issues.append(issue)
                self.results["plugin_bundles"].append(issue or {"file": str(bundle_file), "valid": True})

        return issues

    def _audit_artifact_type_plugin(self, plugin_file: Path) -> dict[str, Any] | None:
        """Audit a single artifact type plugin for schema compliance."""
        try:
            with open(plugin_file) as f:
                plugin_data = yaml.safe_load(f)

            if not plugin_data:
                return {
                    "file": str(plugin_file),
                    "valid": False,
                    "errors": ["Empty or invalid YAML file"],
                }

            errors = []
            warnings = []

            # Check required top-level fields
            for field in self.REQUIRED_PLUGIN_FIELDS["artifact_types"]:
                if field not in plugin_data:
                    errors.append(f"Missing required field: {field}")

            # Check metadata structure
            if "metadata" in plugin_data:
                metadata = plugin_data["metadata"]
                for field in self.REQUIRED_METADATA_FIELDS["artifact_types"]:
                    if field not in metadata:
                        errors.append(f"Missing metadata field: {field}")

                # Validate filename_pattern format
                if "filename_pattern" in metadata:
                    pattern = metadata["filename_pattern"]
                    if not isinstance(pattern, str):
                        errors.append(f"filename_pattern must be string, got {type(pattern)}")
                    elif "{date}" not in pattern:
                        warnings.append("filename_pattern should include {date} placeholder")
                    elif "{name}" not in pattern:
                        warnings.append("filename_pattern should include {name} placeholder")

            # Check validation structure
            if "validation" in plugin_data:
                validation = plugin_data["validation"]
                for field in self.REQUIRED_VALIDATION_FIELDS["artifact_types"]:
                    if field not in validation:
                        errors.append(f"Missing validation field: {field}")

            # Check frontmatter consistency
            if "metadata" in plugin_data and "frontmatter" in plugin_data["metadata"]:
                frontmatter = plugin_data["metadata"]["frontmatter"]
                if "type" not in frontmatter:
                    warnings.append("frontmatter should define 'type' field")

            if errors or warnings:
                return {
                    "file": str(plugin_file),
                    "valid": len(errors) == 0,
                    "errors": errors,
                    "warnings": warnings,
                }

            return None

        except yaml.YAMLError as e:
            return {
                "file": str(plugin_file),
                "valid": False,
                "errors": [f"YAML parsing error: {str(e)}"],
            }
        except Exception as e:
            return {
                "file": str(plugin_file),
                "valid": False,
                "errors": [f"Unexpected error: {str(e)}"],
            }

    def _audit_validators_plugin(self, plugin_file: Path) -> dict[str, Any] | None:
        """Audit the validators plugin for schema compliance."""
        try:
            with open(plugin_file) as f:
                plugin_data = yaml.safe_load(f)

            if not plugin_data:
                return {
                    "file": str(plugin_file),
                    "valid": False,
                    "errors": ["Empty or invalid YAML file"],
                }

            errors = []
            warnings = []

            # Check required top-level fields
            for field in self.REQUIRED_PLUGIN_FIELDS["validators"]:
                if field not in plugin_data:
                    errors.append(f"Missing required field: {field}")

            # Check for expected sections
            if "prefixes" not in plugin_data and "types" not in plugin_data:
                warnings.append("validators.yaml should define custom prefixes or types")

            if errors or warnings:
                return {
                    "file": str(plugin_file),
                    "valid": len(errors) == 0,
                    "errors": errors,
                    "warnings": warnings,
                }

            return None

        except yaml.YAMLError as e:
            return {
                "file": str(plugin_file),
                "valid": False,
                "errors": [f"YAML parsing error: {str(e)}"],
            }
        except Exception as e:
            return {
                "file": str(plugin_file),
                "valid": False,
                "errors": [f"Unexpected error: {str(e)}"],
            }

    def _audit_context_bundle(self, bundle_file: Path) -> dict[str, Any] | None:
        """Audit a context bundle definition for schema compliance."""
        try:
            with open(bundle_file) as f:
                bundle_data = yaml.safe_load(f)

            if not bundle_data:
                return {
                    "file": str(bundle_file),
                    "valid": False,
                    "errors": ["Empty or invalid YAML file"],
                }

            errors = []
            warnings = []

            # Check required top-level fields
            for field in self.REQUIRED_PLUGIN_FIELDS["context_bundles"]:
                if field not in bundle_data:
                    errors.append(f"Missing required field: {field}")

            # Check for files section
            if "files" not in bundle_data:
                errors.append("Missing 'files' section")
            else:
                # Validate file paths exist
                files = bundle_data.get("files", {})
                if isinstance(files, dict):
                    for file_path in files.values():
                        if not (self.project_root / file_path).exists():
                            warnings.append(f"Referenced file not found: {file_path}")

            if errors or warnings:
                return {
                    "file": str(bundle_file),
                    "valid": len(errors) == 0,
                    "errors": errors,
                    "warnings": warnings,
                }

            return None

        except yaml.YAMLError as e:
            return {
                "file": str(bundle_file),
                "valid": False,
                "errors": [f"YAML parsing error: {str(e)}"],
            }
        except Exception as e:
            return {
                "file": str(bundle_file),
                "valid": False,
                "errors": [f"Unexpected error: {str(e)}"],
            }

    def audit_templates(self) -> list[dict[str, Any]]:
        """Audit all artifact templates for schema compliance."""
        issues = []
        templates_dir = self.artifacts_root / "templates"

        if not templates_dir.exists():
            return []

        for template_file in templates_dir.glob("*.md"):
            issue = self._audit_template(template_file)
            if issue:
                issues.append(issue)
            self.results["templates"].append(issue or {"file": str(template_file), "valid": True})

        return issues

    def _audit_template(self, template_file: Path) -> dict[str, Any] | None:
        """Audit a single template file."""
        try:
            with open(template_file) as f:
                content = f.read()

            errors = []
            warnings = []

            # Check for frontmatter
            if not content.strip().startswith("---"):
                errors.append("Template missing frontmatter (should start with ---)")

            # Check for required frontmatter fields in comments or content
            required_placeholders = ["{title}", "{date}", "{type}"]
            for placeholder in required_placeholders:
                if placeholder not in content:
                    warnings.append(f"Template missing expected placeholder: {placeholder}")

            if errors or warnings:
                return {
                    "file": str(template_file),
                    "valid": len(errors) == 0,
                    "errors": errors,
                    "warnings": warnings,
                }

            return None

        except Exception as e:
            return {
                "file": str(template_file),
                "valid": False,
                "errors": [f"Error reading template: {str(e)}"],
            }

    def run_audit(self, check_plugins: bool = True, check_templates: bool = True) -> int:
        """Run full framework audit.

        Args:
            check_plugins: Audit plugins
            check_templates: Audit templates

        Returns:
            Exit code (0 for success, 1 for issues found)
        """
        all_issues = []

        if check_plugins:
            all_issues.extend(self.audit_plugins())

        if check_templates:
            all_issues.extend(self.audit_templates())

        # Update summary
        self.results["summary"]["total_checked"] = sum(
            len(self.results[k]) for k in ["plugin_artifacts", "plugin_validators", "plugin_bundles", "templates"]
        )
        self.results["summary"]["invalid_count"] = len([i for i in all_issues if not i.get("valid", True)])
        self.results["summary"]["warning_count"] = len(
            [w for i in all_issues for w in i.get("warnings", [])]
        )

        return 1 if all_issues else 0

    def generate_report(self) -> str:
        """Generate human-readable audit report."""
        lines = []
        lines.append("=" * 60)
        lines.append("FRAMEWORK AUDIT REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        summary = self.results["summary"]
        lines.append("SUMMARY")
        lines.append("-" * 60)
        lines.append(f"Total Items Checked: {summary['total_checked']}")
        lines.append(f"Invalid Items: {summary['invalid_count']}")
        lines.append(f"Warnings: {summary['warning_count']}")
        lines.append("")

        # Plugin artifacts
        if self.results["plugin_artifacts"]:
            lines.append("ARTIFACT TYPE PLUGINS")
            lines.append("-" * 60)
            for result in self.results["plugin_artifacts"]:
                status = "✓ PASS" if result.get("valid", True) else "✗ FAIL"
                lines.append(f"{status}: {Path(result['file']).name}")
                for error in result.get("errors", []):
                    lines.append(f"  ✗ {error}")
                for warning in result.get("warnings", []):
                    lines.append(f"  ⚠ {warning}")
            lines.append("")

        # Plugin validators
        if self.results["plugin_validators"]:
            lines.append("VALIDATORS PLUGIN")
            lines.append("-" * 60)
            for result in self.results["plugin_validators"]:
                status = "✓ PASS" if result.get("valid", True) else "✗ FAIL"
                lines.append(f"{status}: {Path(result['file']).name}")
                for error in result.get("errors", []):
                    lines.append(f"  ✗ {error}")
                for warning in result.get("warnings", []):
                    lines.append(f"  ⚠ {warning}")
            lines.append("")

        # Plugin bundles
        if self.results["plugin_bundles"]:
            lines.append("CONTEXT BUNDLE PLUGINS")
            lines.append("-" * 60)
            for result in self.results["plugin_bundles"]:
                status = "✓ PASS" if result.get("valid", True) else "✗ FAIL"
                lines.append(f"{status}: {Path(result['file']).name}")
                for error in result.get("errors", []):
                    lines.append(f"  ✗ {error}")
                for warning in result.get("warnings", []):
                    lines.append(f"  ⚠ {warning}")
            lines.append("")

        # Templates
        if self.results["templates"]:
            lines.append("ARTIFACT TEMPLATES")
            lines.append("-" * 60)
            for result in self.results["templates"]:
                status = "✓ PASS" if result.get("valid", True) else "✗ FAIL"
                lines.append(f"{status}: {Path(result['file']).name}")
                for error in result.get("errors", []):
                    lines.append(f"  ✗ {error}")
                for warning in result.get("warnings", []):
                    lines.append(f"  ⚠ {warning}")
            lines.append("")

        lines.append("=" * 60)
        if summary["invalid_count"] == 0:
            lines.append("✓ All items passed validation")
        else:
            lines.append(f"✗ {summary['invalid_count']} items failed validation")
        lines.append("=" * 60)

        return "\n".join(lines)


def main():
    """Main entry point for framework audit."""
    parser = argparse.ArgumentParser(description="Audit AgentQMS framework plugins and templates")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    parser.add_argument(
        "--plugins-only",
        action="store_true",
        help="Audit only plugins (not templates)",
    )
    parser.add_argument(
        "--templates-only",
        action="store_true",
        help="Audit only templates (not plugins)",
    )

    args = parser.parse_args()

    # Determine what to check
    check_plugins = not args.templates_only
    check_templates = not args.plugins_only

    audit = FrameworkAudit(verbose=args.verbose)
    exit_code = audit.run_audit(check_plugins, check_templates)

    if args.json:
        print(json.dumps(audit.results, indent=2))
    else:
        print(audit.generate_report())

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
