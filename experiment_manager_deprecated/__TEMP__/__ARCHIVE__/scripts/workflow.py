#!/usr/bin/env python3
"""
Workflow automation script for experiment tracker.

Provides unified commands for common workflows including:
- Test completion workflow
- Incident report workflow (draft → assess → commit)
- Metadata synchronization
- Smart artifact recording
"""

import argparse
import re
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_manager.core import ExperimentTracker
from experiment_manager.utils.sync import MetadataSync, sync_experiment_metadata


def incident_draft(tracker: ExperimentTracker, experiment_id: str, observations: str, context: str):
    """
    Start incident report workflow - drafting phase.
    Creates a preliminary draft with raw observations.
    """
    print("\n" + "=" * 60)
    print("INCIDENT REPORT - DRAFTING PHASE")
    print("=" * 60)
    print(f"\nExperiment: {experiment_id}")
    print("\nObservations:")
    print(f"  {observations}")
    print("\nContext:")
    print(f"  {context}")
    print("\n" + "=" * 60)
    print("\nNext step: Run 'generate-incident-report.py' to create structured report")
    print("Then run 'workflow.py incident-assess' to evaluate against quality rubric")


def incident_assess(tracker: ExperimentTracker, report_path: str):
    """
    Assess incident report against quality rubric.
    Returns pass/fail with detailed feedback.
    """
    report_file = Path(report_path)
    if not report_file.exists():
        print(f"Error: Report file not found: {report_path}")
        return False

    # Load rubric template
    rubric_path = tracker.root_dir / ".templates" / "incident_report_rubric.md"
    if not rubric_path.exists():
        print(f"Warning: Rubric template not found at {rubric_path}")
        print("Proceeding with basic assessment...")

    print("\n" + "=" * 60)
    print("INCIDENT REPORT - ASSESSMENT PHASE")
    print("=" * 60)
    print(f"\nAssessing: {report_path}")
    print("\nQuality Rubric Check:")
    print("-" * 60)

    # Read report content
    with open(report_file) as f:
        content = f.read()

    # Basic assessment (can be enhanced with AI/ML)
    criteria = {
        "Root Cause Depth": False,
        "Evidence Quality": False,
        "Remediation Logic": False,
        "Metric Impact": False,
    }

    # Check for root cause indicators (not just symptoms)
    if re.search(r"(root cause|why|because|due to|caused by)", content, re.IGNORECASE):
        if not re.search(r"(just|only|simply|tweak|patch)", content, re.IGNORECASE):
            criteria["Root Cause Depth"] = True

    # Check for evidence/artifacts
    if re.search(r"(artifacts?|logs?|images?|screenshots?|see |refer to)", content, re.IGNORECASE):
        criteria["Evidence Quality"] = True

    # Check for remediation/fix
    if re.search(r"(fix|solution|remediation|proposed|address|resolve)", content, re.IGNORECASE):
        criteria["Remediation Logic"] = True

    # Check for metrics/quantification
    if re.search(r"(\d+%|\d+/\d+|success rate|improve|reduce|increase|metric)", content, re.IGNORECASE):
        criteria["Metric Impact"] = True

    # Display results
    all_pass = True
    for criterion, passed in criteria.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {criterion}: {status}")
        if not passed:
            all_pass = False

    print("-" * 60)

    if all_pass:
        print("\n✅ ALL CRITERIA PASS")
        print("Report is ready for committal.")
        print("\nNext step: Run 'workflow.py incident-commit' to finalize")
        return True
    else:
        print("\n❌ ASSESSMENT FAILED")
        print("Report needs revision before committal.")
        print("\nMissing elements:")
        for criterion, passed in criteria.items():
            if not passed:
                print(f"  - {criterion}")
        print("\nSee .templates/incident_report_rubric.md for guidance")
        return False


def incident_commit(tracker: ExperimentTracker, report_path: str, experiment_id: str = None):
    """
    Commit incident report: save, create tasks, link to experiment, track metrics.
    """
    if experiment_id is None:
        experiment_id = tracker._get_current_experiment_id()
        if not experiment_id:
            print("No active experiment found.")
            return False

    report_file = Path(report_path)
    if not report_file.exists():
        print(f"Error: Report file not found: {report_path}")
        return False

    # Ensure report is in incident_reports directory
    paths = tracker._get_paths(experiment_id)
    incident_reports_dir = paths.get_incident_reports_path()

    if not report_file.parent.samefile(incident_reports_dir):
        # Copy to incident_reports directory
        dest_path = incident_reports_dir / report_file.name
        import shutil

        shutil.copy2(report_file, dest_path)
        report_file = dest_path
        print(f"Copied report to: {dest_path}")

    # Record in experiment state
    rel_path = f"incident_reports/{report_file.name}"
    tracker.record_incident_report(rel_path, experiment_id)

    print("\n" + "=" * 60)
    print("INCIDENT REPORT - COMMITTAL PHASE")
    print("=" * 60)
    print(f"\n✅ Report committed to: {experiment_id}")
    print(f"   Path: {rel_path}")
    print("\nNext steps:")
    print("  1. Review the report")
    print("  2. Create task tickets for fixes (use 'add-task.py')")
    print("  3. Track metrics impact when fixes are implemented")

    return True


def record_test_results(tracker: ExperimentTracker, experiment_id: str = None, pattern: str = None):
    """
    Smart artifact recording with auto-detection.
    Finds latest test results and records them.
    """
    if experiment_id is None:
        experiment_id = tracker._get_current_experiment_id()
        if not experiment_id:
            print("No active experiment found.")
            return False

    paths = tracker._get_paths(experiment_id)

    # Try common patterns if not specified
    if pattern is None:
        patterns = [
            "*_worst_performers_test/results.json",
            "*_test_results.json",
            "*results.json",
        ]
    else:
        patterns = [pattern]

    for pattern in patterns:
        artifact_path = paths.find_latest_artifact(pattern)
        if artifact_path:
            print(f"\nFound artifact: {artifact_path.relative_to(paths.base_path)}")
            response = input("Record this artifact? [Y/n]: ").strip().lower()
            if response in ["", "y", "yes"]:
                metadata = {
                    "type": "test_results",
                    "auto_detected": True,
                    "pattern": pattern,
                }
                return tracker.record_artifact(str(artifact_path), metadata, experiment_id, show_context=True, confirm=True)

    print("No matching artifacts found.")
    return False


def sync_metadata(tracker: ExperimentTracker, experiment_id: str = None):
    """
    Manually sync all metadata files.
    """
    if experiment_id is None:
        experiment_id = tracker._get_current_experiment_id()
        if not experiment_id:
            print("No active experiment found.")
            return False

    print(f"\nSyncing metadata for: {experiment_id}")
    result = sync_experiment_metadata(experiment_id, tracker.root_dir, direction="both")

    if result:
        print("✅ Metadata synchronized successfully")

        # Validate consistency
        sync = MetadataSync(experiment_id, tracker.root_dir)
        issues = sync.validate_consistency()
        if issues:
            print("\n⚠️  Consistency issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ No consistency issues found")
    else:
        print("❌ Metadata sync failed")

    return result


def suggest_assessment(tracker: ExperimentTracker, experiment_id: str = None, context: dict = None):
    """
    Suggest creating assessment based on context.
    """
    if experiment_id is None:
        experiment_id = tracker._get_current_experiment_id()
        if not experiment_id:
            return False

    print("\n" + "=" * 60)
    print("ASSESSMENT SUGGESTION")
    print("=" * 60)
    print(f"\nExperiment: {experiment_id}")

    if context and "test_results" in context:
        results = context["test_results"]
        success_rate = results.get("success", 0) / results.get("total", 1) * 100

        if success_rate < 50:
            print("\n💡 Suggestion: Generate 'run-log-negative-result' assessment")
            print("   Low success rate indicates need for failure analysis")
        elif success_rate == 100:
            print("\n💡 Suggestion: Generate 'ab-regression' assessment")
            print("   Perfect results warrant comparison with baseline")
        else:
            print("\n💡 Suggestion: Generate assessment to document findings")

    print("\nRun: generate-assessment.py --template <template_id>")
    return True


def suggest_incident_report(tracker: ExperimentTracker, experiment_id: str = None, context: dict = None):
    """
    Suggest creating incident report based on context (failures, bugs).
    """
    if experiment_id is None:
        experiment_id = tracker._get_current_experiment_id()
        if not experiment_id:
            return False

    print("\n" + "=" * 60)
    print("INCIDENT REPORT SUGGESTION")
    print("=" * 60)
    print(f"\nExperiment: {experiment_id}")

    if context and "failures" in context:
        print("\n💡 Suggestion: Create incident report for failures")
        print("   Run: workflow.py incident-draft")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Workflow automation for experiment tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Incident report workflow
  workflow.py incident-draft --observations "Edge detection failed" --context "Testing worst performers"
  workflow.py incident-assess --report incident_reports/20251129_1200-edge-detection.md
  workflow.py incident-commit --report incident_reports/20251129_1200-edge-detection.md

  # Smart artifact recording
  workflow.py record-test-results

  # Metadata synchronization
  workflow.py sync

  # Suggestions
  workflow.py suggest-assessment
  workflow.py suggest-incident-report
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Incident report commands
    draft_parser = subparsers.add_parser("incident-draft", help="Start incident report (drafting phase)")
    draft_parser.add_argument("--observations", required=True, help="Raw observations")
    draft_parser.add_argument("--context", required=True, help="Context of the failure")

    assess_parser = subparsers.add_parser("incident-assess", help="Assess incident report against rubric")
    assess_parser.add_argument("--report", required=True, help="Path to incident report")

    commit_parser = subparsers.add_parser("incident-commit", help="Commit incident report")
    commit_parser.add_argument("--report", required=True, help="Path to incident report")
    commit_parser.add_argument("--experiment-id", help="Experiment ID (defaults to current)")

    # Other commands
    record_parser = subparsers.add_parser("record-test-results", help="Smart artifact recording")
    record_parser.add_argument("--pattern", help="Pattern to search for (default: auto-detect)")
    record_parser.add_argument("--experiment-id", help="Experiment ID (defaults to current)")

    sync_parser = subparsers.add_parser("sync", help="Sync metadata files")
    sync_parser.add_argument("--experiment-id", help="Experiment ID (defaults to current)")

    suggest_assess_parser = subparsers.add_parser("suggest-assessment", help="Suggest creating assessment")
    suggest_assess_parser.add_argument("--experiment-id", help="Experiment ID (defaults to current)")

    suggest_incident_parser = subparsers.add_parser("suggest-incident-report", help="Suggest creating incident report")
    suggest_incident_parser.add_argument("--experiment-id", help="Experiment ID (defaults to current)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    tracker = ExperimentTracker()

    if args.command == "incident-draft":
        experiment_id = tracker._get_current_experiment_id()
        if not experiment_id:
            print("No active experiment found.")
            sys.exit(1)
        incident_draft(tracker, experiment_id, args.observations, args.context)

    elif args.command == "incident-assess":
        incident_assess(tracker, args.report)

    elif args.command == "incident-commit":
        incident_commit(tracker, args.report, args.experiment_id)

    elif args.command == "record-test-results":
        record_test_results(tracker, args.experiment_id, args.pattern)

    elif args.command == "sync":
        sync_metadata(tracker, args.experiment_id)

    elif args.command == "suggest-assessment":
        suggest_assessment(tracker, args.experiment_id)

    elif args.command == "suggest-incident-report":
        suggest_incident_report(tracker, args.experiment_id)


if __name__ == "__main__":
    main()
