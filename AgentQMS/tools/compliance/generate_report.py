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

from AgentQMS.tools.utils.paths import get_project_root

ROOT = Path(get_project_root())

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def run_command(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    result = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout, result.stderr


def count_files(pattern: str) -> int:
    return len(list(ROOT.glob(pattern)))


def check_ads_compliance() -> dict:
    # Scan specs instead of standards
    spec_files = list((ROOT / "AgentQMS/specs").rglob("*.spec.md"))
    # Also include schemas which might be yaml
    spec_files.extend((ROOT / "AgentQMS/specs").rglob("*.yaml"))

    results = {"total": len(spec_files), "passed": 0, "failed": 0, "warnings": 0, "details": []}

    for spec_file in spec_files:
        # Use our updated validate_artifacts.py which is the new compliance checker
        cmd = ["uv", "run", "python", "AgentQMS/tools/compliance/validate_artifacts.py", "--file", str(spec_file)]
        code, stdout, stderr = run_command(cmd)

        if code == 0:
            results["passed"] += 1
        else:
            results["failed"] += 1
            results["details"].append({"file": spec_file.relative_to(ROOT), "status": "fail", "message": stderr or stdout})

    return results


def check_naming_violations() -> dict:
    violations = []

    for md_file in (ROOT / "docs").glob("*.md"):
        if md_file.name not in ("README.md", "CHANGELOG.md", "CONTRIBUTING.md"):
            if any(c.isupper() for c in md_file.stem.replace("_", "")):
                if md_file.stem.replace("_", "").isupper():
                    violations.append(md_file.relative_to(ROOT))

    return {"total_checked": count_files("docs/*.md"), "violations": len(violations), "files": violations}


def check_placement_violations() -> dict:
    violations = []

    allowed_at_root = {"README.md", "CHANGELOG.md", "CONTRIBUTING.md", "LICENSE"}
    for md_file in (ROOT / "docs").glob("*.md"):
        if md_file.name not in allowed_at_root:
            violations.append(("docs_root", md_file.relative_to(ROOT)))

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
    agents = ["gemini"] # Simplified for now, as tiers changed
    results = {"total_agents": len(agents), "complete": 0, "missing": [], "details": []}

    # Tier 3 is now AgentQMS/specs/tier3-agents/
    # But files are consolidated into agent_identities.spec.md
    # So this check needs to be broader or just check for the spec existence.

    agent_spec = ROOT / "AgentQMS/specs/tier3-agents/agent_identities.spec.md"

    if agent_spec.exists():
        results["complete"] = 1
        results["details"].append({"agent": "gemini", "status": "pass", "missing_files": []})
    else:
         results["missing"].append("gemini")
         results["details"].append({"agent": "gemini", "status": "fail", "missing_files": ["agent_identities.spec.md"]})

    return results


def calculate_token_footprint() -> dict:
    spec_files = list((ROOT / "AgentQMS/specs").rglob("*.spec.md"))
    spec_files.extend((ROOT / "AgentQMS/specs").rglob("*.yaml"))

    total_lines = 0
    tier_breakdown = defaultdict(int)

    for spec_file in spec_files:
        lines = len(spec_file.read_text().splitlines())
        total_lines += lines

        if "tier1" in str(spec_file):
            tier_breakdown["tier1"] += lines
        elif "tier2" in str(spec_file):
            tier_breakdown["tier2"] += lines
        elif "tier3" in str(spec_file):
            tier_breakdown["tier3"] += lines
        elif "tier4" in str(spec_file):
            tier_breakdown["tier4"] += lines
        elif "schema" in str(spec_file):
            tier_breakdown["schema"] += lines

    estimated_tokens = total_lines * 4


    return {
        "total_lines": total_lines,
        "estimated_tokens": estimated_tokens,
        "tier_breakdown": dict(tier_breakdown),
        "files_analyzed": len(spec_files),
    }


def generate_report() -> str:
    print(f"{BLUE}🔍 Generating AI Documentation Compliance Dashboard...{RESET}\n")

    ads_compliance = check_ads_compliance()
    naming = check_naming_violations()
    placement = check_placement_violations()
    agents = check_agent_configs()
    footprint = calculate_token_footprint()

    report = f"""
{'=' * 80}
AI DOCUMENTATION COMPLIANCE DASHBOARD
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}

📊 OVERALL STATUS
├─ ADS v1.0 Compliance: {ads_compliance['passed']}/{ads_compliance['total']} files pass
├─ Naming Violations: {naming['violations']} files
├─ Placement Violations: {placement['total_violations']} files
├─ Agent Configs: {agents['complete']}/{agents['total_agents']} complete
└─ Token Footprint: ~{footprint['estimated_tokens']:,} tokens ({footprint['total_lines']:,} lines)

{'─' * 80}

✅ ADS v1.0 COMPLIANCE
├─ Total YAML Files: {ads_compliance['total']}
├─ Passed: {GREEN}{ads_compliance['passed']}{RESET}
├─ Failed: {RED if ads_compliance['failed'] > 0 else GREEN}{ads_compliance['failed']}{RESET}
└─ Warnings: {YELLOW if ads_compliance['warnings'] > 0 else GREEN}{ads_compliance['warnings']}{RESET}
"""

    if ads_compliance["failed"] > 0:
        report += "\n   ❌ Failed Files:\n"
        for detail in ads_compliance["details"]:
            if detail["status"] == "fail":
                report += f"      - {detail['file']}\n"

    report += f"""
{'─' * 80}

📛 NAMING VIOLATIONS
├─ Files Checked: {naming['total_checked']}
├─ ALL-CAPS Violations: {RED if naming['violations'] > 0 else GREEN}{naming['violations']}{RESET}
"""

    if naming["violations"] > 0:
        report += "   ❌ Violations Found:\n"
        for viol in naming["files"]:
            report += f"      - {viol}\n"
    else:
        report += f"   {GREEN}✓ No naming violations{RESET}\n"

    report += f"""
{'─' * 80}

📂 PLACEMENT VIOLATIONS
├─ Total Violations: {RED if placement['total_violations'] > 0 else GREEN}{placement['total_violations']}{RESET}
├─ Files at docs/ root: {len(placement['docs_root'])}
└─ Files at docs/artifacts/ root: {len(placement['artifacts_root'])}
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
{'─' * 80}

🤖 AGENT CONFIGURATIONS
├─ Total Agents: {agents['total_agents']}
├─ Complete Configs: {GREEN if agents['complete'] == agents['total_agents'] else YELLOW}{agents['complete']}{RESET}
└─ Missing Configs: {RED if agents['missing'] else GREEN}{len(agents['missing'])}{RESET}

   Status by Agent:
"""

    for detail in agents["details"]:
        status_icon = "✓" if detail["status"] == "pass" else "✗"
        status_color = GREEN if detail["status"] == "pass" else RED
        report += f"   {status_color}{status_icon}{RESET} {detail['agent']}: {detail['status']}\n"
        if detail["missing_files"]:
            report += f"      Missing: {', '.join(detail['missing_files'])}\n"

    report += f"""
{'─' * 80}

💾 TOKEN FOOTPRINT ANALYSIS
├─ Total Lines: {footprint['total_lines']:,}
├─ Estimated Tokens: ~{footprint['estimated_tokens']:,}
└─ Files Analyzed: {footprint['files_analyzed']}

   Breakdown by Tier:
"""

    for tier, lines in sorted(footprint["tier_breakdown"].items()):
        tokens = lines * 4
        report += f"   ├─ {tier}: {lines} lines (~{tokens} tokens)\n"

    total_checks = 4
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
{'─' * 80}

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


def main() -> None:
    try:
        report = generate_report()
        print(report)

        report_file = ROOT / "docs" / "artifacts" / "compliance_reports" / "latest-report.txt"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(report)

        print(f"\n📄 Report saved to: {report_file.relative_to(ROOT)}\n")

    except Exception as exc:
        print(f"\n{RED}❌ Error generating report: {exc}{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
