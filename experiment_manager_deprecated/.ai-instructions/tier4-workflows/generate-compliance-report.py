#!/usr/bin/env python3
"""
EDS v1.0 Compliance Dashboard Generator

Generates comprehensive compliance report for all experiments.
Identifies violations, calculates compliance metrics, prioritizes remediation.

Usage:
    python generate-compliance-report.py
    python generate-compliance-report.py --experiment 20251217_024343_image_enhancements
    python generate-compliance-report.py --output compliance-reports/report.md
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_compliance_checker(file_path: Path, checker_path: Path) -> tuple[bool, list[str], list[str]]:
    """
    Run compliance checker on a file.

    Returns:
        (is_valid, errors, warnings)
    """
    try:
        result = subprocess.run(["python3", str(checker_path), str(file_path)], capture_output=True, text=True, timeout=10)

        output = result.stdout + result.stderr
        is_valid = result.returncode == 0

        # Parse errors and warnings from output
        errors = []
        warnings = []

        for line in output.split("\n"):
            if "• " in line or "Missing" in line or "Invalid" in line:
                errors.append(line.strip())
            elif "⚠️" in line:
                warnings.append(line.strip())

        return is_valid, errors, warnings

    except subprocess.TimeoutExpired:
        return False, ["Validation timeout"], []
    except Exception as e:
        return False, [f"Validation error: {str(e)}"], []


def analyze_experiment(experiment_dir: Path, checker_path: Path) -> dict:
    """
    Analyze a single experiment for EDS v1.0 compliance.

    Returns:
        {
            'experiment_id': str,
            'artifacts': List[Dict],
            'total_count': int,
            'compliant_count': int,
            'violation_count': int,
            'critical_violations': List[str],
            'compliance_percentage': float,
            'has_metadata_dir': bool,
            'all_caps_count': int,
            'missing_frontmatter_count': int,
        }
    """
    experiment_id = experiment_dir.name
    artifacts = []
    compliant_count = 0
    violation_count = 0
    critical_violations = []
    all_caps_count = 0
    missing_frontmatter_count = 0

    # Check for .metadata/ directory
    has_metadata_dir = (experiment_dir / ".metadata").exists()

    # Find all markdown files
    md_files = list(experiment_dir.rglob("*.md"))

    for md_file in md_files:
        # Skip README.md
        if md_file.name == "README.md":
            continue

        filename = md_file.name
        relative_path = md_file.relative_to(experiment_dir)

        # Check for ALL-CAPS
        is_all_caps = filename.replace(".md", "").replace("_", "").isupper()
        if is_all_caps:
            all_caps_count += 1

        # Run compliance checker
        is_valid, errors, warnings = run_compliance_checker(md_file, checker_path)

        # Check for missing frontmatter
        has_missing_frontmatter = any("No YAML frontmatter" in err for err in errors)
        if has_missing_frontmatter:
            missing_frontmatter_count += 1

        artifact_info = {
            "filename": filename,
            "relative_path": str(relative_path),
            "is_valid": is_valid,
            "is_all_caps": is_all_caps,
            "errors": errors,
            "warnings": warnings,
        }

        artifacts.append(artifact_info)

        if is_valid:
            compliant_count += 1
        else:
            violation_count += 1

            # Identify critical violations
            if is_all_caps:
                critical_violations.append(f"ALL-CAPS filename: {filename}")
            if has_missing_frontmatter:
                critical_violations.append(f"Missing frontmatter: {filename}")

    total_count = len(artifacts)
    compliance_percentage = (compliant_count / total_count * 100) if total_count > 0 else 0

    return {
        "experiment_id": experiment_id,
        "artifacts": artifacts,
        "total_count": total_count,
        "compliant_count": compliant_count,
        "violation_count": violation_count,
        "critical_violations": critical_violations,
        "compliance_percentage": compliance_percentage,
        "has_metadata_dir": has_metadata_dir,
        "all_caps_count": all_caps_count,
        "missing_frontmatter_count": missing_frontmatter_count,
    }


def generate_markdown_report(experiments_data: list[dict], output_path: Path):
    """Generate markdown compliance report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Sort experiments by compliance percentage (worst first)
    experiments_data.sort(key=lambda x: x["compliance_percentage"])

    # Calculate aggregate metrics
    total_experiments = len(experiments_data)
    total_artifacts = sum(e["total_count"] for e in experiments_data)
    total_compliant = sum(e["compliant_count"] for e in experiments_data)
    total_violations = sum(e["violation_count"] for e in experiments_data)
    avg_compliance = sum(e["compliance_percentage"] for e in experiments_data) / total_experiments if total_experiments > 0 else 0

    total_all_caps = sum(e["all_caps_count"] for e in experiments_data)
    total_missing_frontmatter = sum(e["missing_frontmatter_count"] for e in experiments_data)
    experiments_without_metadata = sum(1 for e in experiments_data if not e["has_metadata_dir"])

    # Calculate percentages safely
    compliant_pct = (total_compliant / total_artifacts * 100) if total_artifacts > 0 else 0
    violation_pct = (total_violations / total_artifacts * 100) if total_artifacts > 0 else 0
    all_caps_pct = (total_all_caps / total_artifacts * 100) if total_artifacts > 0 else 0
    missing_fm_pct = (total_missing_frontmatter / total_artifacts * 100) if total_artifacts > 0 else 0
    missing_meta_pct = (experiments_without_metadata / total_experiments * 100) if total_experiments > 0 else 0

    # Generate report
    report = f"""# EDS v1.0 Compliance Report

**Generated**: {timestamp}

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Experiments | {total_experiments} |
| Total Artifacts | {total_artifacts} |
| Compliant Artifacts | {total_compliant} ({compliant_pct:.1f}%) |
| Violations | {total_violations} ({violation_pct:.1f}%) |
| Average Compliance | {avg_compliance:.1f}% |

## Critical Violations Summary

| Violation Type | Count | Percentage |
|----------------|-------|------------|
| ALL-CAPS Filenames | {total_all_caps} | {all_caps_pct:.1f}% |
| Missing Frontmatter | {total_missing_frontmatter} | {missing_fm_pct:.1f}% |
| Missing .metadata/ Directory | {experiments_without_metadata} | {missing_meta_pct:.1f}% |

## Experiment Compliance (Worst First)

| Experiment ID | Artifacts | Compliant | Violations | Compliance | .metadata/ | ALL-CAPS | Missing FM |
|---------------|-----------|-----------|------------|------------|------------|----------|------------|
"""

    for exp in experiments_data:
        metadata_status = "✅" if exp["has_metadata_dir"] else "❌"
        report += f"| {exp['experiment_id']} | {exp['total_count']} | {exp['compliant_count']} | {exp['violation_count']} | {exp['compliance_percentage']:.1f}% | {metadata_status} | {exp['all_caps_count']} | {exp['missing_frontmatter_count']} |\n"

    # Detailed violations per experiment
    report += "\n## Detailed Violations\n\n"

    for exp in experiments_data:
        if exp["violation_count"] == 0:
            continue

        report += f"### {exp['experiment_id']}\n\n"
        report += f"**Compliance**: {exp['compliance_percentage']:.1f}% ({exp['compliant_count']}/{exp['total_count']} artifacts)\n\n"

        if exp["critical_violations"]:
            report += "**Critical Violations**:\n\n"
            for violation in exp["critical_violations"]:
                report += f"- ❌ {violation}\n"
            report += "\n"

        # List non-compliant artifacts
        non_compliant_artifacts = [a for a in exp["artifacts"] if not a["is_valid"]]
        if non_compliant_artifacts:
            report += "**Non-Compliant Artifacts**:\n\n"
            for artifact in non_compliant_artifacts:
                report += f"- `{artifact['relative_path']}`\n"
                for error in artifact["errors"][:3]:  # Limit to 3 errors per file
                    report += f"  - {error}\n"
            report += "\n"

    # Remediation priorities
    report += "\n## Remediation Priorities\n\n"

    # Priority 1: Experiments without .metadata/
    if experiments_without_metadata > 0:
        report += "### Priority 1: Missing .metadata/ Directories\n\n"
        report += f"**Count**: {experiments_without_metadata} experiments\n\n"
        report += "**Action**: Create .metadata/ directory structure:\n\n"
        report += "```bash\n"
        for exp in experiments_data:
            if not exp["has_metadata_dir"]:
                report += f"mkdir -p experiment_manager/experiments/{exp['experiment_id']}/.metadata/{{assessments,reports,guides,scripts,artifacts}}\n"
        report += "```\n\n"

    # Priority 2: ALL-CAPS filenames
    if total_all_caps > 0:
        report += "### Priority 2: ALL-CAPS Filenames\n\n"
        report += f"**Count**: {total_all_caps} files\n\n"
        report += "**Action**: Rename using lowercase-hyphenated pattern:\n\n"
        for exp in experiments_data:
            all_caps_artifacts = [a for a in exp["artifacts"] if a["is_all_caps"]]
            if all_caps_artifacts:
                report += f"**{exp['experiment_id']}**:\n"
                for artifact in all_caps_artifacts:
                    old_name = artifact["filename"]
                    # Suggest new name (simplified example)
                    new_name = f"20251217_1200_guide_{old_name.lower().replace('_', '-').replace('.md', '')}.md"
                    report += f"- `{old_name}` → `{new_name}`\n"
                report += "\n"

    # Priority 3: Missing frontmatter
    if total_missing_frontmatter > 0:
        report += "### Priority 3: Missing Frontmatter\n\n"
        report += f"**Count**: {total_missing_frontmatter} files\n\n"
        report += "**Action**: Add EDS v1.0 compliant frontmatter using CLI tools:\n\n"
        report += "```bash\n"
        report += "# Use CLI to regenerate artifacts with correct frontmatter\n"
        report += "# eds generate-assessment --experiment <id> --slug <slug>\n"
        report += "# eds generate-report --experiment <id> --slug <slug>\n"
        report += "```\n\n"

    # Write report
    output_path.write_text(report)
    print(f"✅ Compliance report generated: {output_path}")
    print(f"\n📊 Summary: {avg_compliance:.1f}% average compliance ({total_compliant}/{total_artifacts} artifacts)")


def main():
    parser = argparse.ArgumentParser(description="Generate EDS v1.0 compliance report")
    parser.add_argument("--experiment", help="Analyze specific experiment ID")
    parser.add_argument("--output", help="Output report path", default=None)
    args = parser.parse_args()

    # Paths
    script_dir = Path(__file__).parent
    ai_instructions_dir = script_dir.parent
    tracker_dir = ai_instructions_dir.parent
    experiments_dir = tracker_dir / "experiments"
    checker_path = ai_instructions_dir / "schema" / "compliance-checker.py"

    if not checker_path.exists():
        print(f"❌ ERROR: Compliance checker not found: {checker_path}")
        sys.exit(1)

    if not experiments_dir.exists():
        print(f"❌ ERROR: Experiments directory not found: {experiments_dir}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = Path(__file__).parent / "compliance-reports"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"compliance-report-{timestamp}.md"

    print("🔍 EDS v1.0 Compliance Analysis")
    print(f"📂 Experiments directory: {experiments_dir}")
    print(f"📝 Output: {output_path}")
    print("")

    # Collect experiments to analyze
    if args.experiment:
        experiment_dirs = [experiments_dir / args.experiment]
        if not experiment_dirs[0].exists():
            print(f"❌ ERROR: Experiment not found: {args.experiment}")
            sys.exit(1)
    else:
        experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]

    print(f"📊 Analyzing {len(experiment_dirs)} experiment(s)...\n")

    # Analyze experiments
    experiments_data = []
    for exp_dir in sorted(experiment_dirs):
        print(f"   Analyzing: {exp_dir.name}...")
        exp_data = analyze_experiment(exp_dir, checker_path)
        experiments_data.append(exp_data)

    print("")

    # Generate report
    generate_markdown_report(experiments_data, output_path)

    # Print summary to console
    print("\n" + "=" * 60)
    print("Experiments by Compliance:")
    for exp in sorted(experiments_data, key=lambda x: x["compliance_percentage"]):
        status = "✅" if exp["compliance_percentage"] == 100 else "❌"
        print(f"  {status} {exp['experiment_id']:<50} {exp['compliance_percentage']:>5.1f}%")

    # Print critical issues
    total_critical = sum(len(e["critical_violations"]) for e in experiments_data)
    if total_critical > 0:
        print("\n" + "=" * 60)
        print(f"⚠️  {total_critical} critical violations require immediate attention")
        print(f"📖 See detailed report: {output_path}")


if __name__ == "__main__":
    main()
