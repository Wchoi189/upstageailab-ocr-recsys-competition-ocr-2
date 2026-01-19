#!/usr/bin/env python3
"""
Systematic Configuration Standards Audit for ocr/ Module

This script audits the ocr/ module for compliance with:
AgentQMS/standards/tier2-framework/configuration-standards.yaml

It uses:
1. Existing AST analysis outputs (config_access.txt)
2. Live AST scanning via agent-debug-toolkit
3. Direct grep patterns for common violations
"""

import json
import re
import subprocess
from pathlib import Path
from collections import defaultdict

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
OCR_MODULE = PROJECT_ROOT / "ocr"
CONFIG_ACCESS_FILE = PROJECT_ROOT / "project_compass" / "config_access.txt"


class ConfigComplianceAuditor:
    """Audit configuration handling compliance."""

    def __init__(self):
        self.violations: dict[str, list[dict]] = defaultdict(list)
        self.stats = {
            "files_scanned": 0,
            "total_violations": 0,
            "critical_violations": 0,
        }

    def load_existing_scan(self) -> dict:
        """Load existing config_access.txt analysis."""
        if not CONFIG_ACCESS_FILE.exists():
            print(f"⚠️  {CONFIG_ACCESS_FILE} not found. Run AST scan first.")
            return {}

        try:
            with open(CONFIG_ACCESS_FILE) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse {CONFIG_ACCESS_FILE}: {e}")
            return {}

    def check_isinstance_dict_violations(self, scan_data: dict) -> None:
        """
        Rule: no-isinstance-dict (CRITICAL)
        NEVER use isinstance(obj, dict) on config objects.
        """
        if not scan_data or "results" not in scan_data:
            return

        for entry in scan_data["results"]:
            snippet = entry.get("code_snippet", "")
            file_path = entry.get("file", "")

            # Check for isinstance(x, dict) without OmegaConf.is_dict
            if re.search(r'isinstance\([^,]+,\s*dict\)', snippet):
                # Exclude cases that also check DictConfig (allowed)
                if "DictConfig" not in snippet:
                    self.violations["no-isinstance-dict"].append({
                        "file": file_path,
                        "line": entry.get("line"),
                        "snippet": snippet.strip(),
                        "severity": "CRITICAL",
                        "rule": "no-isinstance-dict",
                        "fix": "Use is_config() from ocr.core.utils.config_utils"
                    })

    def check_dict_conversion_violations(self, scan_data: dict) -> None:
        """
        Rule: use-ensure-dict (CRITICAL)
        ALWAYS use ensure_dict() instead of dict() or OmegaConf.to_container()
        """
        if not scan_data or "results" not in scan_data:
            return

        bad_patterns = [
            (r'dict\([^)]*cfg[^)]*\)', "dict(cfg)"),
            (r'OmegaConf\.to_container', "OmegaConf.to_container()"),
        ]

        for entry in scan_data["results"]:
            snippet = entry.get("code_snippet", "")
            file_path = entry.get("file", "")

            for pattern, desc in bad_patterns:
                if re.search(pattern, snippet):
                    # Skip if ensure_dict is already being used
                    if "ensure_dict" not in snippet:
                        self.violations["use-ensure-dict"].append({
                            "file": file_path,
                            "line": entry.get("line"),
                            "snippet": snippet.strip(),
                            "severity": "CRITICAL",
                            "rule": "use-ensure-dict",
                            "pattern": desc,
                            "fix": "Use ensure_dict() from ocr.core.utils.config_utils"
                        })

    def scan_with_grep(self) -> None:
        """Fallback: direct grep for common violations."""
        patterns = [
            ("isinstance.*dict", "isinstance check"),
            (r"dict\(.*cfg", "dict() conversion"),
            ("OmegaConf.to_container", "to_container() usage"),
        ]

        for pattern, desc in patterns:
            try:
                result = subprocess.run(
                    ["grep", "-rn", "-E", pattern, str(OCR_MODULE), "--include=*.py"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                if result.returncode == 0:
                    lines = result.stdout.strip().split("\n")
                    print(f"  Found {len(lines)} occurrences of '{desc}'")

                    # Sample first 5 for quick review
                    for line in lines[:5]:
                        if line:
                            file_line, code = line.split(":", 1)
                            print(f"    {file_line}: {code[:80]}")
            except subprocess.TimeoutExpired:
                print(f"  Grep timeout for pattern '{pattern}'")
            except Exception as e:
                print(f"  Grep failed for '{pattern}': {e}")

    def check_imports(self) -> dict[str, int]:
        """Check if files use the proper utilities."""
        result = subprocess.run(
            ["grep", "-r", "from ocr.core.utils.config_utils import",
             str(OCR_MODULE), "--include=*.py"],
            capture_output=True,
            text=True
        )

        files_with_imports = set()
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line:
                    file_path = line.split(":")[0]
                    files_with_imports.add(file_path)

        return {
            "files_importing_utils": len(files_with_imports),
            "files": list(files_with_imports)[:10]  # Sample
        }

    def generate_report(self) -> str:
        """Generate audit report."""
        report = []
        report.append("=" * 80)
        report.append("Configuration Standards Compliance Audit")
        report.append("Module: ocr/")
        report.append("Standard: AgentQMS/standards/tier2-framework/configuration-standards.yaml")
        report.append("=" * 80)
        report.append("")

        # Summary
        total_violations = sum(len(v) for v in self.violations.values())
        report.append("📊 Summary:")
        report.append(f"  Total Violations: {total_violations}")
        report.append(f"  Critical Rules Violated: {len(self.violations)}")
        report.append("")

        # Violations by rule
        for rule_id, violations in self.violations.items():
            report.append(f"🚨 Rule: {rule_id} ({len(violations)} violations)")
            report.append("")

            # Group by file
            by_file = defaultdict(list)
            for v in violations:
                by_file[v["file"]].append(v)

            for file_path, file_violations in sorted(by_file.items())[:10]:  # Top 10 files
                rel_path = Path(file_path).relative_to(PROJECT_ROOT)
                report.append(f"  📄 {rel_path} ({len(file_violations)} violations)")

                for v in file_violations[:3]:  # First 3 per file
                    report.append(f"     Line {v['line']}: {v['snippet'][:100]}")
                    report.append(f"     💡 Fix: {v['fix']}")
                    report.append("")

            if len(by_file) > 10:
                report.append(f"  ... and {len(by_file) - 10} more files")
            report.append("")

        return "\n".join(report)

    def run_audit(self) -> None:
        """Execute full audit."""
        print("🔍 Starting Configuration Standards Audit...")
        print()

        # Phase 1: Load existing scan
        print("📂 Phase 1: Loading existing AST scan data...")
        scan_data = self.load_existing_scan()
        if scan_data:
            print(f"  ✅ Loaded {len(scan_data.get('results', []))} entries")
        else:
            print("  ⚠️  No existing scan data")
        print()

        # Phase 2: Check rules
        print("🔎 Phase 2: Checking compliance rules...")
        print("  • Checking isinstance(dict) violations...")
        self.check_isinstance_dict_violations(scan_data)
        print(f"    Found: {len(self.violations['no-isinstance-dict'])} violations")

        print("  • Checking dict conversion violations...")
        self.check_dict_conversion_violations(scan_data)
        print(f"    Found: {len(self.violations['use-ensure-dict'])} violations")
        print()

        # Phase 3: Grep fallback
        print("🔍 Phase 3: Running grep pattern scan...")
        self.scan_with_grep()
        print()

        # Phase 4: Check imports
        print("📦 Phase 4: Checking utility imports...")
        import_stats = self.check_imports()
        print(f"  ✅ {import_stats['files_importing_utils']} files import config_utils")
        print()

        # Generate report
        print("📝 Generating report...")
        report = self.generate_report()

        # Save report
        output_file = PROJECT_ROOT / "docs" / "reports" / "config_compliance_audit.txt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(report)

        print(f"✅ Report saved to: {output_file}")
        print()
        print(report)


def main():
    auditor = ConfigComplianceAuditor()
    auditor.run_audit()


if __name__ == "__main__":
    main()
