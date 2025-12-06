#!/usr/bin/env python3
"""
Documentation Quality Monitor
Automatically detects and reports documentation issues that agents encounter
"""

import re
from datetime import datetime
from pathlib import Path


class DocumentationQualityMonitor:
    """Monitors documentation quality and detects issues."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.issues = []

    def check_documentation_consistency(self) -> list[dict]:
        """Check for documentation consistency issues."""
        issues = []

        # Check for conflicting instructions
        instruction_files = [
            "docs/AI_AGENT_SYSTEM.md",
            ".cursor/rules/prompts-artifacts-guidelines.mdc",
            "docs/AI_AGENT_GUIDE.md",
            "docs/AI_AGENT_GUIDELINES.md",
            ".ai_instructions",
        ]

        for file_path in instruction_files:
            if Path(file_path).exists():
                issues.extend(self._check_file_consistency(file_path))

        return issues

    def check_tool_paths(self) -> list[dict]:
        """Check for outdated tool path references."""
        issues = []

        # Patterns that indicate outdated paths
        outdated_patterns = [
            r"scripts/agent_tools/artifact_workflow\.py",
            r"scripts/agent_tools/validate_artifacts\.py",
            r"scripts/agent_tools/monitor_artifacts\.py",
        ]

        # Files to check
        check_files = [
            "docs/AI_AGENT_SYSTEM.md",
            ".cursor/rules/prompts-artifacts-guidelines.mdc",
            "docs/artifacts/templates/2025-01-27_1700_template-ai-agent-automation-tools-usage.md",
        ]

        for file_path in check_files:
            if Path(file_path).exists():
                content = Path(file_path).read_text()
                for pattern in outdated_patterns:
                    matches = re.findall(pattern, content)
                    if matches:
                        issues.append(
                            {
                                "type": "outdated_path",
                                "file": file_path,
                                "issue": f"Outdated path reference: {pattern}",
                                "severity": "high",
                                "suggested_fix": "Update to use new organized paths (core/, compliance/, etc.)",
                            }
                        )

        return issues

    def check_deprecated_files(self) -> list[dict]:
        """Check for deprecated files that should be removed."""
        issues = []

        deprecated_files = [
            "docs/AI_AGENT_GUIDE.md",
            "docs/AI_AGENT_GUIDELINES.md",
            "docs/ai_handbook/ai-agent-entry-point.md",
            ".ai_instructions",
        ]

        for file_path in deprecated_files:
            if Path(file_path).exists():
                # Check if it has proper deprecation notice
                content = Path(file_path).read_text()
                if "DEPRECATED" not in content.upper():
                    issues.append(
                        {
                            "type": "missing_deprecation_notice",
                            "file": file_path,
                            "issue": "File exists but lacks deprecation notice",
                            "severity": "medium",
                            "suggested_fix": "Add deprecation notice pointing to docs/AI_AGENT_SYSTEM.md",
                        }
                    )

        return issues

    def check_tool_discoverability(self) -> list[dict]:
        """Check if tools are discoverable."""
        issues = []

        # Check if discovery script exists
        discovery_script = Path("scripts/agent_tools/discover.py")
        if not discovery_script.exists():
            issues.append(
                {
                    "type": "missing_discovery",
                    "file": "scripts/agent_tools/discover.py",
                    "issue": "Tool discovery script missing",
                    "severity": "high",
                    "suggested_fix": "Create discovery script to help agents find tools",
                }
            )

        # Check if wrapper scripts exist
        wrapper_scripts = [
            "create-artifact",
            "validate-artifacts",
            "monitor-compliance",
            "agent-help",
        ]

        for script in wrapper_scripts:
            if not Path(script).exists():
                issues.append(
                    {
                        "type": "missing_wrapper",
                        "file": script,
                        "issue": f"Wrapper script {script} missing",
                        "severity": "medium",
                        "suggested_fix": f"Create wrapper script {script} for easier tool access",
                    }
                )

        return issues

    def check_documentation_completeness(self) -> list[dict]:
        """Check if documentation is complete."""
        issues = []

        # Check if main system doc has all required sections
        main_doc = Path("docs/AI_AGENT_SYSTEM.md")
        if main_doc.exists():
            content = main_doc.read_text()
            required_sections = [
                "Tool Discovery",
                "Available Automation Tools",
                "Easy Access Commands",
            ]

            for section in required_sections:
                if section not in content:
                    issues.append(
                        {
                            "type": "missing_section",
                            "file": str(main_doc),
                            "issue": f"Missing section: {section}",
                            "severity": "medium",
                            "suggested_fix": f"Add {section} section to main documentation",
                        }
                    )

        return issues

    def generate_quality_report(self) -> str:
        """Generate a comprehensive quality report."""
        all_issues = []
        all_issues.extend(self.check_documentation_consistency())
        all_issues.extend(self.check_tool_paths())
        all_issues.extend(self.check_deprecated_files())
        all_issues.extend(self.check_tool_discoverability())
        all_issues.extend(self.check_documentation_completeness())

        report = []
        report.append("# Documentation Quality Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Group by severity
        high_issues = [i for i in all_issues if i["severity"] == "high"]
        medium_issues = [i for i in all_issues if i["severity"] == "medium"]
        low_issues = [i for i in all_issues if i["severity"] == "low"]

        if high_issues:
            report.append("## 🚨 High Priority Issues")
            for issue in high_issues:
                report.append(f"### {issue['type'].replace('_', ' ').title()}")
                report.append(f"**File**: {issue['file']}")
                report.append(f"**Issue**: {issue['issue']}")
                report.append(f"**Suggested Fix**: {issue['suggested_fix']}")
                report.append("")

        if medium_issues:
            report.append("## ⚠️ Medium Priority Issues")
            for issue in medium_issues:
                report.append(f"### {issue['type'].replace('_', ' ').title()}")
                report.append(f"**File**: {issue['file']}")
                report.append(f"**Issue**: {issue['issue']}")
                report.append(f"**Suggested Fix**: {issue['suggested_fix']}")
                report.append("")

        if low_issues:
            report.append("## ℹ️ Low Priority Issues")
            for issue in low_issues:
                report.append(f"### {issue['type'].replace('_', ' ').title()}")
                report.append(f"**File**: {issue['file']}")
                report.append(f"**Issue**: {issue['issue']}")
                report.append(f"**Suggested Fix**: {issue['suggested_fix']}")
                report.append("")

        # Summary
        report.append("## 📊 Summary")
        report.append(f"- **Total Issues**: {len(all_issues)}")
        report.append(f"- **High Priority**: {len(high_issues)}")
        report.append(f"- **Medium Priority**: {len(medium_issues)}")
        report.append(f"- **Low Priority**: {len(low_issues)}")

        return "\n".join(report)

    def _check_file_consistency(self, file_path: str) -> list[dict]:
        """Check a specific file for consistency issues."""
        issues = []

        try:
            content = Path(file_path).read_text()

            # Check for conflicting tool paths
            if (
                "scripts/agent_tools/artifact_workflow.py" in content
                and "scripts/agent_tools/core/artifact_workflow.py" in content
            ):
                issues.append(
                    {
                        "type": "conflicting_paths",
                        "file": file_path,
                        "issue": "Contains both old and new tool paths",
                        "severity": "high",
                        "suggested_fix": "Update all references to use new organized paths",
                    }
                )

            # Check for deprecated references
            if "AI_AGENT_GUIDE.md" in content or "AI_AGENT_GUIDELINES.md" in content:
                issues.append(
                    {
                        "type": "deprecated_reference",
                        "file": file_path,
                        "issue": "References deprecated documentation files",
                        "severity": "medium",
                        "suggested_fix": "Update references to point to docs/AI_AGENT_SYSTEM.md",
                    }
                )

        except Exception as e:
            issues.append(
                {
                    "type": "file_error",
                    "file": file_path,
                    "issue": f"Error reading file: {e!s}",
                    "severity": "medium",
                    "suggested_fix": "Check file permissions and format",
                }
            )

        return issues


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Documentation Quality Monitor")
    parser.add_argument("--check", action="store_true", help="Run quality checks")
    parser.add_argument("--report", action="store_true", help="Generate quality report")
    parser.add_argument("--output", help="Output file for report")

    args = parser.parse_args()

    monitor = DocumentationQualityMonitor()

    if args.check or args.report:
        report = monitor.generate_quality_report()

        if args.output:
            Path(args.output).write_text(report)
            print(f"Quality report saved to {args.output}")
        else:
            print(report)
    else:
        print("Use --help for usage information")


if __name__ == "__main__":
    main()
