#!/usr/bin/env python3
"""
Migration Auditor Script - Hydra Configuration Structural Debt Detector

Purpose: Identify violations of Hydra architectural laws before refactoring
Usage:   uv run python 02_SCRIPT_migration_auditor.py [--config-root configs]
Output:  refactor_audit_report.txt with categorized violations

This script detects:
1. Missing @package directives
2. UI/Frontend bloat in training configs
3. Domain leakage (cross-domain key contamination)
4. Legacy files in production directories
"""

import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


class ViolationType(Enum):
    """Categories of configuration violations"""
    MISSING_PACKAGE = "MISSING HEADER"
    UI_BLOAT = "UI BLOAT"
    DOMAIN_LEAKAGE = "LEAKAGE"
    LEGACY_POLLUTION = "LEGACY POLLUTION"
    PACKAGE_ERROR = "PACKAGE ERROR"


@dataclass
class Violation:
    """Represents a single configuration violation"""
    type: ViolationType
    path: Path
    message: str
    severity: str = "WARNING"  # WARNING, ERROR, CRITICAL

    def __str__(self):
        icon = "🚩" if self.severity == "CRITICAL" else "⚠️"
        return f"{icon} [{self.type.value}] {self.path}: {self.message}"


class ConfigAuditor:
    """Audits Hydra configuration files for structural violations"""

    # Keywords indicating UI/Frontend bloat
    UI_BLOAT_KEYWORDS = ["frontend", "ui", "dashboard", "preset", "profile"]

    # Domain-specific keys that indicate leakage
    DOMAIN_KEYS = {
        "detection": ["max_polygons", "shrink_ratio", "thresh_min", "thresh_max"],
        "recognition": ["max_label_length", "charset", "case_sensitive"],
        "kie": ["max_entities", "relation_types"]
    }

    def __init__(self, config_root: str = "configs"):
        self.config_root = Path(config_root)
        self.violations: list[Violation] = []

    def audit_all(self) -> list[Violation]:
        """Run all audit checks on configuration directory"""
        print(f"🔍 Auditing configuration directory: {self.config_root}")

        for root, dirs, files in os.walk(self.config_root):
            # Skip legacy directories
            if "__LEGACY__" in root or "__EXTENDED__" in root:
                self._check_legacy_location(Path(root))
                continue

            for file in files:
                if not file.endswith((".yaml", ".yml")):
                    continue

                path = Path(root) / file
                self._audit_file(path)

        return self.violations

    def _audit_file(self, path: Path):
        """Audit a single configuration file"""
        try:
            content = path.read_text()

            # Check 1: Package directive presence
            self._check_package_directive(path, content)

            # Check 2: UI/Frontend bloat
            self._check_ui_bloat(path, content)

            # Check 3: Domain leakage
            self._check_domain_leakage(path, content)

            # Check 4: Package placement errors
            self._check_package_placement(path, content)

        except Exception as e:
            self.violations.append(Violation(
                type=ViolationType.PACKAGE_ERROR,
                path=path,
                message=f"Failed to parse: {e}",
                severity="ERROR"
            ))

    def _check_package_directive(self, path: Path, content: str):
        """Verify @package directive is present"""
        if "@package" not in content:
            # Exception: config.yaml doesn't need @package
            if path.name == "config.yaml":
                return

            self.violations.append(Violation(
                type=ViolationType.MISSING_PACKAGE,
                path=path,
                message="No @package directive found",
                severity="CRITICAL"
            ))

    def _check_ui_bloat(self, path: Path, content: str):
        """Detect UI/Frontend configs in training tree"""
        content_lower = content.lower()

        if any(keyword in content_lower for keyword in self.UI_BLOAT_KEYWORDS):
            # Check if in training/logger path
            if "training/logger" in str(path):
                self.violations.append(Violation(
                    type=ViolationType.UI_BLOAT,
                    path=path,
                    message="Contains UI-related logic in training tree. Should be in archive/ui_configs/",
                    severity="CRITICAL"
                ))

    def _check_domain_leakage(self, path: Path, content: str):
        """Detect cross-domain key contamination"""
        # Determine which domain this file belongs to
        path_str = str(path)
        current_domain = None

        for domain in self.DOMAIN_KEYS.keys():
            if f"/{domain}/" in path_str or f"/domain/{domain}" in path_str:
                current_domain = domain
                break

        if not current_domain:
            return  # Not a domain-specific file

        # Check for keys from OTHER domains
        for other_domain, keys in self.DOMAIN_KEYS.items():
            if other_domain == current_domain:
                continue

            for key in keys:
                if key in content:
                    # Check if it's explicitly nullified (acceptable)
                    if f"{key}: null" in content or f"{key}:null" in content:
                        continue

                    self.violations.append(Violation(
                        type=ViolationType.DOMAIN_LEAKAGE,
                        path=path,
                        message=f"{other_domain.capitalize()} key '{key}' found in {current_domain} config",
                        severity="CRITICAL"
                    ))

    def _check_package_placement(self, path: Path, content: str):
        """Verify @package directive matches file location"""
        if "@package _global_" in content:
            # Should only be in hardware/ or experiment/
            if not any(x in str(path) for x in ["hardware", "experiment", "global"]):
                self.violations.append(Violation(
                    type=ViolationType.PACKAGE_ERROR,
                    path=path,
                    message="Uses @package _global_ but not in hardware/experiment/global directory",
                    severity="ERROR"
                ))

        elif "@package _group_" in content:
            # Should be in component directories
            if not any(x in str(path) for x in ["model", "data", "train", "optimizer", "scheduler"]):
                self.violations.append(Violation(
                    type=ViolationType.PACKAGE_ERROR,
                    path=path,
                    message="Uses @package _group_ but not in component directory",
                    severity="WARNING"
                ))

    def _check_legacy_location(self, path: Path):
        """Detect legacy directories still in configs/"""
        if self.config_root in path.parents:
            self.violations.append(Violation(
                type=ViolationType.LEGACY_POLLUTION,
                path=path,
                message="Legacy directory found in configs/. Should be moved to archive/",
                severity="CRITICAL"
            ))

    def generate_report(self, output_file: str = "refactor_audit_report.txt"):
        """Generate categorized violation report"""
        # Group by severity
        critical = [v for v in self.violations if v.severity == "CRITICAL"]
        errors = [v for v in self.violations if v.severity == "ERROR"]
        warnings = [v for v in self.violations if v.severity == "WARNING"]

        with open(output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("HYDRA CONFIGURATION AUDIT REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Total Violations: {len(self.violations)}\n")
            f.write(f"  - Critical: {len(critical)}\n")
            f.write(f"  - Errors: {len(errors)}\n")
            f.write(f"  - Warnings: {len(warnings)}\n\n")

            # Critical violations first
            if critical:
                f.write("\n" + "=" * 80 + "\n")
                f.write("CRITICAL VIOLATIONS (Must Fix Before Refactor)\n")
                f.write("=" * 80 + "\n")
                for v in critical:
                    f.write(f"{v}\n")

            # Then errors
            if errors:
                f.write("\n" + "=" * 80 + "\n")
                f.write("ERRORS (Should Fix)\n")
                f.write("=" * 80 + "\n")
                for v in errors:
                    f.write(f"{v}\n")

            # Finally warnings
            if warnings:
                f.write("\n" + "=" * 80 + "\n")
                f.write("WARNINGS (Review Recommended)\n")
                f.write("=" * 80 + "\n")
                for v in warnings:
                    f.write(f"{v}\n")

        print(f"\n✅ Audit complete. Found {len(self.violations)} issues.")
        print(f"📄 Report saved to: {output_file}")

        return output_file


def main():
    """Run the configuration audit"""
    import argparse

    parser = argparse.ArgumentParser(description="Audit Hydra configuration for structural violations")
    parser.add_argument("--config-root", default="configs", help="Root configuration directory")
    parser.add_argument("--output", default="refactor_audit_report.txt", help="Output report file")

    args = parser.parse_args()

    auditor = ConfigAuditor(config_root=args.config_root)
    violations = auditor.audit_all()
    auditor.generate_report(output_file=args.output)

    # Exit with error code if critical violations found
    critical_count = sum(1 for v in violations if v.severity == "CRITICAL")
    if critical_count > 0:
        print(f"\n❌ Found {critical_count} critical violations. Refactor cannot proceed safely.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
