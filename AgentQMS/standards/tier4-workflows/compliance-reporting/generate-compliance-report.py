#!/usr/bin/env python3
"""
AI Documentation Compliance Dashboard
Generates comprehensive compliance reports for ADS v1.0 and AgentQMS artifacts
"""

import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def run_command(cmd: list[str], cwd=None) -> tuple[int, str, str]:
    """Run shell command and return exit code, stdout, stderr"""
    result = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout, result.stderr


def count_files(pattern: str) -> int:
    """Count files matching glob pattern"""
    return len(list(ROOT.glob(pattern)))


def check_ads_compliance() -> dict:
    """Check ADS v1.0 compliance for all YAML files"""
    yaml_files = list(Path(ROOT / ".ai-instructions").rglob("*.yaml"))

    results = {"total": len(yaml_files), "passed": 0, "failed": 0, "warnings": 0, "details": []}

    for yaml_file in yaml_files:
        cmd = ["python3", ".ai-instructions/schema/compliance-checker.py", str(yaml_file)]
        code, stdout, stderr = run_command(cmd)

        if code == 0:
            results["passed"] += 1
            if "⚠️" in stdout:
                results["warnings"] += 1
                results["details"].append(
                    {"file": yaml_file.relative_to(ROOT), "status": "pass_with_warnings", "message": "Contains user-oriented phrases"}
                )
        else:
            results["failed"] += 1
            results["details"].append({"file": yaml_file.relative_to(ROOT), "status": "fail", "message": stderr or stdout})

    return results


def check_naming_violations() -> dict:
    """Check for ALL-CAPS filenames at docs/ root"""
    violations = []

    for md_file in (ROOT / "docs").glob("*.md"):
        if md_file.name not in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md"):
            # Check if filename has ALL-CAPS pattern
            if any(c.isupper() for c in md_file.stem.replace("_", "")):
                if md_file.stem.replace("_", "").isupper():
                    violations.append(md_file.relative_to(ROOT))

    return {"total_checked": count_files("docs/*.md"), "violations": len(violations), "files": violations}


def check_placement_violations() -> dict:
    """Check for misplaced artifacts"""
    violations = []

    # Check docs/ root (except allowed files)
    allowed_at_root = {"README.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE"}
    for md_file in (ROOT / "docs").glob("*.md"):
        if md_file.name not in allowed_at_root:
            violations.append(("docs_root", md_file.relative_to(ROOT)))

    # Check docs/artifacts/ root (should be in subdirectories)
    artifacts_dir = ROOT / "docs" / "artifacts"
    if artifacts_dir.exists():
        for md_file in artifacts_dir.glob("*.md"):
            if md_file.name != "INDEX.md":
                violations.append(("artifacts_root", md_file.relative_to(ROOT)))

    return {
        "total_violations": len(violations),
        "docs_root": [f for loc, f in violations if loc == "docs_root"],
        "artifacts_root": [f for loc, f in violations if loc == "artifacts_root"],
    }


def check_agent_configs() -> dict:
    """Check all agent configurations exist and validate"""
    agents = ["claude", "copilot", "cursor", "gemini"]
    results = {"total_agents": len(agents), "complete": 0, "missing": [], "details": []}

    for agent in agents:
        agent_dir = ROOT / ".ai-instructions" / "tier3-agents" / agent
        required_files = ["config.yaml", "quick-reference.yaml", "validation.sh"]

        missing = []
        for req_file in required_files:
            if not (agent_dir / req_file).exists():
                missing.append(req_file)

        if not missing:
            results["complete"] += 1
            # Run validation script
            val_script = agent_dir / "validation.sh"
            if val_script.exists():
                code, stdout, stderr = run_command(["bash", str(val_script)])
                status = "pass" if code == 0 else "fail"
            else:
                status = "no_validation"
        else:
            results["missing"].append(agent)
            status = "incomplete"

        results["details"].append({"agent": agent, "status": status, "missing_files": missing})

    return results


def calculate_token_footprint() -> dict:
    """Estimate token footprint of AI documentation"""
    yaml_files = list(Path(ROOT / ".ai-instructions").rglob("*.yaml"))

    total_lines = 0
    tier_breakdown = defaultdict(int)

    for yaml_file in yaml_files:
        lines = len(yaml_file.read_text().splitlines())
        total_lines += lines

        # Determine tier
        if "tier1-sst" in str(yaml_file):
            tier_breakdown["tier1"] += lines
        elif "tier2-framework" in str(yaml_file):
            tier_breakdown["tier2"] += lines
        elif "tier3-agents" in str(yaml_file):
            tier_breakdown["tier3"] += lines
        elif "tier4-workflows" in str(yaml_file):
            tier_breakdown["tier4"] += lines
        elif "schema" in str(yaml_file):
            tier_breakdown["schema"] += lines

    # Rough estimate: 1 line YAML ≈ 4 tokens
    estimated_tokens = total_lines * 4

    return {
        "total_lines": total_lines,
        "estimated_tokens": estimated_tokens,
        "tier_breakdown": dict(tier_breakdown),
        "files_analyzed": len(yaml_files),
    }


def generate_report() -> str:
    """Generate comprehensive compliance report"""
    print(f"{BLUE}🔍 Generating AI Documentation Compliance Dashboard...{RESET}\n")

    # Gather data
    ads_compliance = check_ads_compliance()
    naming = check_naming_violations()
    placement = check_placement_violations()
    agents = check_agent_configs()
    footprint = calculate_token_footprint()

    # Build report
    report = f"""
{"=" * 80}
AI DOCUMENTATION COMPLIANCE DASHBOARD
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{"=" * 80}

📊 OVERALL STATUS
├─ ADS v1.0 Compliance: {ads_compliance["passed"]}/{ads_compliance["total"]} files pass
├─ Naming Violations: {naming["violations"]} files
├─ Placement Violations: {placement["total_violations"]} files
├─ Agent Configs: {agents["complete"]}/{agents["total_agents"]} complete
└─ Token Footprint: ~{footprint["estimated_tokens"]:,} tokens ({footprint["total_lines"]:,} lines)

{"─" * 80}

✅ ADS v1.0 COMPLIANCE
├─ Total YAML Files: {ads_compliance["total"]}
├─ Passed: {GREEN}{ads_compliance["passed"]}{RESET}
├─ Failed: {RED if ads_compliance["failed"] > 0 else GREEN}{ads_compliance["failed"]}{RESET}
└─ Warnings: {YELLOW if ads_compliance["warnings"] > 0 else GREEN}{ads_compliance["warnings"]}{RESET}
"""

    if ads_compliance["failed"] > 0:
        report += "\n   ❌ Failed Files:\n"
        for detail in ads_compliance["details"]:
            if detail["status"] == "fail":
                report += f"      - {detail['file']}\n"

    report += f"""
{"─" * 80}

📛 NAMING VIOLATIONS
├─ Files Checked: {naming["total_checked"]}
├─ ALL-CAPS Violations: {RED if naming["violations"] > 0 else GREEN}{naming["violations"]}{RESET}
"""

    if naming["violations"] > 0:
        report += "   ❌ Violations Found:\n"
        for viol in naming["files"]:
            report += f"      - {viol}\n"
    else:
        report += f"   {GREEN}✓ No naming violations{RESET}\n"

    report += f"""
{"─" * 80}

📂 PLACEMENT VIOLATIONS
├─ Total Violations: {RED if placement["total_violations"] > 0 else GREEN}{placement["total_violations"]}{RESET}
├─ Files at docs/ root: {len(placement["docs_root"])}
└─ Files at docs/artifacts/ root: {len(placement["artifacts_root"])}
"""

    if placement["total_violations"] > 0:
        if placement["docs_root"]:
            report += "\n   ❌ docs/ root violations:\n"
            for viol in placement["docs_root"]:
                report += f"      - {viol}\n"
        if placement["artifacts_root"]:
            report += "\n   ❌ docs/artifacts/ root violations:\n"
            for viol in placement["artifacts_root"]:
                report += f"      - {viol}\n"
    else:
        report += f"   {GREEN}✓ No placement violations{RESET}\n"

    report += f"""
{"─" * 80}

🤖 AGENT CONFIGURATIONS
├─ Total Agents: {agents["total_agents"]}
├─ Complete Configs: {GREEN if agents["complete"] == agents["total_agents"] else YELLOW}{agents["complete"]}{RESET}
└─ Missing Configs: {RED if agents["missing"] else GREEN}{len(agents["missing"])}{RESET}

   Status by Agent:
"""

    for detail in agents["details"]:
        status_icon = "✓" if detail["status"] == "pass" else "✗"
        status_color = GREEN if detail["status"] == "pass" else RED
        report += f"   {status_color}{status_icon}{RESET} {detail['agent']}: {detail['status']}\n"
        if detail["missing_files"]:
            report += f"      Missing: {', '.join(detail['missing_files'])}\n"

    report += f"""
{"─" * 80}

💾 TOKEN FOOTPRINT ANALYSIS
├─ Total Lines: {footprint["total_lines"]:,}
├─ Estimated Tokens: ~{footprint["estimated_tokens"]:,}
└─ Files Analyzed: {footprint["files_analyzed"]}

   Breakdown by Tier:
"""

    for tier, lines in sorted(footprint["tier_breakdown"].items()):
        tokens = lines * 4
        report += f"   ├─ {tier}: {lines} lines (~{tokens} tokens)\n"

    # Overall compliance score
    total_checks = 4  # ADS, naming, placement, agents
    passed_checks = 0

    if ads_compliance["failed"] == 0:
        passed_checks += 1
    if naming["violations"] == 0:
        passed_checks += 1
    if placement["total_violations"] == 0:
        passed_checks += 1
    if agents["complete"] == agents["total_agents"]:
        passed_checks += 1

    compliance_percent = (passed_checks / total_checks) * 100

    report += f"""
{"─" * 80}

🎯 COMPLIANCE SCORE: {compliance_percent:.0f}% ({passed_checks}/{total_checks} checks passed)

"""

    if compliance_percent == 100:
        report += f"{GREEN}✅ FULL COMPLIANCE ACHIEVED{RESET}\n"
    elif compliance_percent >= 75:
        report += f"{YELLOW}⚠️  MOSTLY COMPLIANT - Minor issues to address{RESET}\n"
    else:
        report += f"{RED}❌ COMPLIANCE ISSUES DETECTED - Action required{RESET}\n"

    report += f"\n{'=' * 80}\n"

    return report


def main():
    """Main entry point"""
    try:
        report = generate_report()
        print(report)

        # Write to file
        report_file = ROOT / ".ai-instructions" / "tier4-workflows" / "compliance-reporting" / "latest-report.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report)

        print(f"\n📄 Report saved to: {report_file.relative_to(ROOT)}\n")

    except Exception as e:
        print(f"\n{RED}❌ Error generating report: {e}{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
