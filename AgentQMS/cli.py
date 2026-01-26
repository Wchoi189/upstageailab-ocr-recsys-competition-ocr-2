#!/usr/bin/env python3
"""
AgentQMS Unified CLI Tool

Consolidated entry point for all AgentQMS tools with subcommand structure.
Replaces 5+ separate scripts with a single unified interface.

Usage:
    aqms artifact create --type implementation_plan --name "my-feature" --title "My Feature"
    aqms validate --file path/to/artifact.md
    aqms monitor --check
    aqms feedback report --issue-type "documentation" --description "Issue description"
    aqms quality check
    aqms generate-config --path ocr/inference
    aqms check-infra

Available Commands:
    artifact          Artifact workflow management (create, validate, update)
    validate          Validate artifacts and compliance
    monitor           Monitor artifact organization and compliance
    feedback          Collect agent feedback and suggestions
    quality           Documentation quality monitoring
    generate-config   Generate effective.yaml with path-aware discovery
    check-infra       Verify connectivity to Redis, RabbitMQ, and Ollama
"""

import argparse
import sys
from pathlib import Path

# Ensure paths are set up correctly using framework utilities
try:
    # Try importing from installed package first
    from AgentQMS.tools.utils.paths import get_project_root
    # We don't have ensure_project_root_on_sys_path in the file I read,
    # but we can implement similar logic or just use get_project_root return value
    project_root = get_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except ImportError:
    # Fallback for development/pre-install: find root manually
    current_path = Path(__file__).resolve()
    # Go up from AgentQMS/cli.py to project root
    project_root = current_path.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def setup_artifact_parser(subparsers):
    """Setup artifact workflow subcommand."""
    parser = subparsers.add_parser(
        "artifact",
        help="Artifact workflow management",
        description="Create, validate, and manage artifacts",
    )

    artifact_subparsers = parser.add_subparsers(dest="artifact_command", help="Artifact commands")

    # artifact create
    create_parser = artifact_subparsers.add_parser("create", help="Create a new artifact")
    create_parser.add_argument("--type", required=True, help="Artifact type")
    create_parser.add_argument("--name", required=True, help="Artifact name (kebab-case)")
    create_parser.add_argument("--title", required=True, help="Artifact title")
    create_parser.add_argument("--no-validate", action="store_true", help="Skip auto-validation")
    create_parser.add_argument("--no-indexes", action="store_true", help="Skip index updates")
    create_parser.add_argument("--no-track", action="store_true", help="Skip tracking registration")
    create_parser.add_argument("--content", help="Initial content (overwrites template)")

    # artifact validate
    validate_parser = artifact_subparsers.add_parser("validate", help="Validate artifact(s)")
    validate_parser.add_argument("--file", help="Validate specific file")
    validate_parser.add_argument("--all", action="store_true", help="Validate all artifacts")

    # artifact update-indexes
    artifact_subparsers.add_parser("update-indexes", help="Update artifact indexes")

    # artifact check-compliance
    artifact_subparsers.add_parser("check-compliance", help="Run compliance checks")

    return parser


def setup_validate_parser(subparsers):
    """Setup validation subcommand."""
    parser = subparsers.add_parser(
        "validate",
        help="Validate artifacts and compliance",
        description="Run validation checks on artifacts",
    )

    parser.add_argument("--file", help="Validate specific file")
    parser.add_argument("--directory", help="Validate directory")
    parser.add_argument("--all", action="store_true", help="Validate all artifacts")
    parser.add_argument("--check-naming", action="store_true", help="Check naming conventions only")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--quiet", action="store_true", help="Suppress output except errors")

    return parser


def setup_monitor_parser(subparsers):
    """Setup monitoring subcommand."""
    parser = subparsers.add_parser(
        "monitor",
        help="Monitor artifact organization and compliance",
        description="Monitor compliance and provide alerts",
    )

    parser.add_argument("--check", action="store_true", help="Run compliance check")
    parser.add_argument("--alert", action="store_true", help="Generate alerts for violations")
    parser.add_argument("--report", action="store_true", help="Generate compliance report")
    parser.add_argument("--fix-suggestions", action="store_true", help="Show fix suggestions")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    return parser


def setup_feedback_parser(subparsers):
    """Setup feedback collection subcommand."""
    parser = subparsers.add_parser(
        "feedback",
        help="Collect agent feedback and suggestions",
        description="Report issues and suggest improvements",
    )

    feedback_subparsers = parser.add_subparsers(dest="feedback_command", help="Feedback commands")

    # feedback report
    report_parser = feedback_subparsers.add_parser("report", help="Report an issue")
    report_parser.add_argument("--issue-type", required=True, help="Type of issue")
    report_parser.add_argument("--description", required=True, help="Issue description")
    report_parser.add_argument("--file-path", help="Related file path")
    report_parser.add_argument("--severity", default="medium", choices=["low", "medium", "high", "critical"], help="Issue severity")
    report_parser.add_argument("--suggested-fix", help="Suggested fix")
    report_parser.add_argument("--context", help="Additional context")

    # feedback suggest
    suggest_parser = feedback_subparsers.add_parser("suggest", help="Suggest an improvement")
    suggest_parser.add_argument("--area", required=True, help="Area for improvement")
    suggest_parser.add_argument("--current", required=True, help="Current state")
    suggest_parser.add_argument("--suggested", required=True, help="Suggested change")
    suggest_parser.add_argument("--rationale", required=True, help="Rationale for change")
    suggest_parser.add_argument("--priority", default="medium", choices=["low", "medium", "high"], help="Priority")

    # feedback list
    list_parser = feedback_subparsers.add_parser("list", help="List feedback items")
    list_parser.add_argument("--status", choices=["open", "closed", "all"], default="all", help="Filter by status")
    list_parser.add_argument("--type", help="Filter by type")

    return parser


def setup_quality_parser(subparsers):
    """Setup quality monitoring subcommand."""
    parser = subparsers.add_parser(
        "quality",
        help="Documentation quality monitoring",
        description="Monitor and check documentation quality",
    )

    parser.add_argument("--check", action="store_true", help="Run quality checks")
    parser.add_argument("--consistency", action="store_true", help="Check documentation consistency")
    parser.add_argument("--tool-paths", action="store_true", help="Check for outdated tool paths")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    return parser


def setup_generate_config_parser(subparsers):
    """Setup generate-config subcommand."""
    parser = subparsers.add_parser(
        "generate-config",
        help="Generate effective.yaml with path-aware discovery",
        description="Generate effective configuration with dynamic standard discovery",
    )

    parser.add_argument("--path", help="Current working path for standard discovery")
    parser.add_argument("--output", default="AgentQMS/.agentqms/effective.yaml", help="Output path")
    parser.add_argument("--registry", default="AgentQMS/standards/registry.yaml", help="Registry path")
    parser.add_argument("--settings", default="AgentQMS/.agentqms/settings.yaml", help="Settings path")
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout instead of writing")
    parser.add_argument("--json", action="store_true", help="Output in JSON format (Virtual Mode)")

    return parser


def run_artifact_command(args):
    """Execute artifact subcommand."""
    from AgentQMS.tools.core.artifact_workflow import ArtifactWorkflow

    workflow = ArtifactWorkflow()

    if args.artifact_command == "create":
        result = workflow.create_artifact(
            artifact_type=args.type,
            name=args.name,
            title=args.title,
            auto_validate=not args.no_validate,
            auto_update_indexes=not args.no_indexes,
            track=not args.no_track,
            content=args.content,
        )
        print(f"✅ Created artifact: {result}")
        return 0

    elif args.artifact_command == "validate":
        if args.file:
            success = workflow.validator.validate_file(Path(args.file))
            return 0 if success else 1
        elif args.all:
            success = workflow.validator.validate_all()
            return 0 if success else 1
        else:
            print("Error: Specify --file or --all")
            return 1

    elif args.artifact_command == "update-indexes":
        workflow.update_indexes()
        print("✅ Indexes updated")
        return 0

    elif args.artifact_command == "check-compliance":
        workflow.check_compliance()
        return 0

    else:
        print("Error: Unknown artifact command")
        return 1


def run_validate_command(args):
    """Execute validate subcommand."""
    from AgentQMS.tools.compliance.validate_artifacts import ArtifactValidator

    validator = ArtifactValidator()

    if args.file:
        results = validator.validate_single_file(Path(args.file))
        if results["valid"]:
            print(f"✅ {args.file}: Valid")
            return 0
        else:
            print(f"❌ {args.file}: Invalid")
            for error in results.get("errors", []):
                print(f"  - {error}")
            return 1
    elif args.directory:
        results = validator.validate_directory(Path(args.directory))
        valid_count = sum(1 for r in results if r["valid"])
        total_count = len(results)
        print(f"✅ Validated {valid_count}/{total_count} artifacts in {args.directory}")
        if valid_count < total_count:
            print(f"❌ {total_count - valid_count} artifacts have errors")
            return 1
        return 0
    elif args.all:
        results = validator.validate_all()
        valid_count = sum(1 for r in results if r["valid"])
        total_count = len(results)
        print(f"✅ Validated {valid_count}/{total_count} artifacts")
        if valid_count < total_count:
            print(f"❌ {total_count - valid_count} artifacts have errors")
            for result in results:
                if not result["valid"]:
                    print(f"\n{result['file']}:")
                    for error in result.get("errors", []):
                        print(f"  - {error}")
            return 1
        return 0
    elif args.check_naming:
        success = validator.check_naming_conventions()
        return 0 if success else 1
    else:
        print("Error: Specify --file, --directory, --all, or --check-naming")
        return 1


def run_monitor_command(args):
    """Execute monitor subcommand."""
    from AgentQMS.tools.compliance.monitor_artifacts import ArtifactMonitor

    monitor = ArtifactMonitor()

    if args.check:
        results = monitor.check_organization_compliance()
        if args.json:
            import json
            print(json.dumps(results, indent=2))
        else:
            report_text = monitor.generate_compliance_report(results)
            print(report_text)
        return 0 if results.get("compliance_rate", 0) >= 95 else 1

    elif args.alert:
        alerts = monitor.generate_alerts()
        if args.json:
            import json
            print(json.dumps(alerts, indent=2))
        else:
            for alert in alerts:
                print(f"⚠️  {alert['message']}")
        return 0

    elif args.report:
        report = monitor.generate_report()
        print(report)
        return 0

    elif args.fix_suggestions:
        suggestions = monitor.get_fix_suggestions()
        for suggestion in suggestions:
            print(f"💡 {suggestion}")
        return 0

    else:
        print("Error: Specify --check, --alert, --report, or --fix-suggestions")
        return 1


def run_feedback_command(args):
    """Execute feedback subcommand."""
    from AgentQMS.tools.utilities.agent_feedback import AgentFeedbackCollector

    collector = AgentFeedbackCollector()

    if args.feedback_command == "report":
        feedback_id = collector.collect_feedback(
            issue_type=args.issue_type,
            description=args.description,
            file_path=args.file_path,
            severity=args.severity,
            suggested_fix=args.suggested_fix,
            agent_context=args.context,
        )
        print(f"✅ Feedback recorded: {feedback_id}")
        return 0

    elif args.feedback_command == "suggest":
        suggestion_id = collector.suggest_improvement(
            area=args.area,
            current_state=args.current,
            suggested_change=args.suggested,
            rationale=args.rationale,
            priority=args.priority,
        )
        print(f"✅ Suggestion recorded: {suggestion_id}")
        return 0

    elif args.feedback_command == "list":
        items = collector.list_feedback(status=args.status, issue_type=args.type)
        for item in items:
            print(f"[{item['id']}] {item['description']}")
        return 0

    else:
        print("Error: Unknown feedback command")
        return 1


def run_quality_command(args):
    """Execute quality subcommand."""
    from AgentQMS.tools.compliance.documentation_quality_monitor import DocumentationQualityMonitor

    monitor = DocumentationQualityMonitor()

    if args.check or args.consistency:
        issues = monitor.check_documentation_consistency()
        if args.json:
            import json
            print(json.dumps(issues, indent=2))
        else:
            for issue in issues:
                print(f"⚠️  {issue['message']}")
        return 0 if not issues else 1

    elif args.tool_paths:
        issues = monitor.check_tool_paths()
        if args.json:
            import json
            print(json.dumps(issues, indent=2))
        else:
            for issue in issues:
                print(f"⚠️  {issue['message']}")
        return 0 if not issues else 1

    else:
        # Run all checks
        consistency_issues = monitor.check_documentation_consistency()
        path_issues = monitor.check_tool_paths()
        all_issues = consistency_issues + path_issues

        if args.json:
            import json
            print(json.dumps(all_issues, indent=2))
        else:
            print(f"Found {len(all_issues)} quality issues")
            for issue in all_issues:
                print(f"⚠️  {issue['message']}")

        return 0 if not all_issues else 1


def run_generate_config_command(args):
    """Execute generate-config subcommand."""
    from AgentQMS.tools.utils.config_loader import ConfigLoader
    import json

    try:
        import yaml
    except ImportError:
        # PyYAML is optional if we are just outputting JSON
        if not args.json:
            print("Error: PyYAML not available. Install with: pip install pyyaml or use --json")
            return 1

    loader = ConfigLoader()

    # Generate virtual effective config
    effective = loader.generate_virtual_config(
        settings_path=args.settings,
        registry_path=args.registry,
        current_path=args.path,
    )

    # Add tool_mappings from settings if present
    settings = loader.get_config(args.settings, defaults={})
    if "tool_mappings" in settings.get("resolved", {}):
        effective["resolved"]["tool_mappings"] = settings["resolved"]["tool_mappings"]

    if args.json:
        # Direct output for AI ingestion (No file created)
        print(json.dumps(effective))
        return 0
    
    # For YAML output (legacy or explicit file generation)
    if "yaml" in vars() or "yaml" in globals():
        yaml_output = yaml.dump(effective, sort_keys=False, default_flow_style=False)
    else:
        # Fallback if yaml not imported (should be caught above, but safety first)
        import yaml
        yaml_output = yaml.dump(effective, sort_keys=False, default_flow_style=False)

    if args.dry_run:
        print("=" * 60)
        print("Generated effective.yaml (dry-run):")
        print("=" * 60)
        print(yaml_output)
        print("=" * 60)
        active_standards = effective["resolved"]["context_integration"].get("active_standards", [])
        print(f"\nActive Standards ({len(active_standards)}):")
        for std in active_standards:
            print(f"  - {std}")
    else:
        # Only write if output path is explicitly provided or we are in legacy mode
        # For now, we preserve writing to the default path if not --json, 
        # but the plan says "Stop physically writing...". 
        # However, the args.output has a default value in usage. 
        # We will respect that for now to avoid breaking existing workflows completely,
        # but --json is the preferred AI way.
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(yaml_output)

        print(f"✅ Generated: {output_path}")
        print(f"📁 Current path: {args.path or 'current directory'}")

        active_standards = effective["resolved"]["context_integration"].get("active_standards", [])
        print(f"📋 Active standards: {len(active_standards)}")

        if active_standards:
            print("\nActive Standards:")
            for std in active_standards[:5]:
                print(f"  - {std}")
            if len(active_standards) > 5:
                print(f"  ... and {len(active_standards) - 5} more")

    return 0


def setup_registry_parser(subparsers):
    """Setup registry management subcommand."""
    parser = subparsers.add_parser(
        "registry",
        help="ADS v2.0 Registry management",
        description="Manage standards registry, resolve dependencies, and migrate standards",
    )

    registry_subparsers = parser.add_subparsers(dest="registry_command", help="Registry commands")

    # registry sync
    sync_parser = registry_subparsers.add_parser("sync", help="Compile registry from ADS headers")
    sync_parser.add_argument("--dry-run", action="store_true", help="Validate without writing")
    sync_parser.add_argument("--strict", action="store_true", help="Strict mode: fail if no standards")
    sync_parser.add_argument("--no-graph", action="store_true", help="Skip DOT graph generation")

    # registry resolve
    resolve_parser = registry_subparsers.add_parser("resolve", help="Resolve standards by task/path/keywords")
    resolve_parser.add_argument("--task", help="Resolve by task type")
    resolve_parser.add_argument("--path", help="Resolve by file path")
    resolve_parser.add_argument("--keywords", help="Resolve by keywords (space-separated)")
    resolve_parser.add_argument("--query", help="Shorthand for --keywords")
    resolve_parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching")
    resolve_parser.add_argument("--threshold", type=float, default=0.8, help="Fuzzy threshold (0.0-1.0)")
    resolve_parser.add_argument("--no-deps", action="store_true", help="Don't expand dependencies")
    resolve_parser.add_argument("--no-tier1", action="store_true", help="Don't include Tier 1 standards")
    resolve_parser.add_argument("--paths-only", action="store_true", help="Output only file paths")
    resolve_parser.add_argument("--no-cache", action="store_true", help="Disable binary cache")
    resolve_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # registry suggest-header
    suggest_parser = registry_subparsers.add_parser("suggest-header", help="Suggest ADS v2.0 header for legacy standard")
    suggest_parser.add_argument("file", help="Legacy standard file path")
    suggest_parser.add_argument("--apply", action="store_true", help="Apply suggested header")
    suggest_parser.add_argument("--force", action="store_true", help="Force overwrite existing file")

    # registry validate
    validate_parser = registry_subparsers.add_parser("validate", help="Validate ADS v2.0 compliance")
    validate_parser.add_argument("files", nargs="*", help="Files to validate (default: all)")
    validate_parser.add_argument("--strict", action="store_true", help="Strict validation mode")

    return parser


def setup_check_infra_parser(subparsers):
    """Setup check-infra subcommand."""
    parser = subparsers.add_parser(
        "check-infra",
        help="Verify connectivity to Redis, RabbitMQ, and Ollama",
        description="Run preflight checks on infrastructure services",
    )
    return parser


def run_registry_command(args):
    """Execute registry subcommand."""
    import subprocess
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    if args.registry_command == "sync":
        # Run sync_registry.py
        cmd = ["uv", "run", "python", str(project_root / "AgentQMS" / "tools" / "sync_registry.py")]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.strict:
            cmd.append("--strict")
        if args.no_graph:
            cmd.append("--no-graph")

        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode

    elif args.registry_command == "resolve":
        # Run resolve_standards.py
        cmd = ["uv", "run", "python", str(project_root / "AgentQMS" / "tools" / "resolve_standards.py")]

        if args.task:
            cmd.extend(["--task", args.task])
        if args.path:
            cmd.extend(["--path", args.path])
        if args.keywords:
            cmd.extend(["--keywords", args.keywords])
        if args.query:
            cmd.extend(["--query", args.query])
        if args.fuzzy:
            cmd.append("--fuzzy")
        if args.threshold != 0.8:
            cmd.extend(["--threshold", str(args.threshold)])
        if args.no_deps:
            cmd.append("--no-deps")
        if args.no_tier1:
            cmd.append("--no-tier1")
        if args.paths_only:
            cmd.append("--paths-only")
        if args.no_cache:
            cmd.append("--no-cache")
        if args.verbose:
            cmd.append("-v")

        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode

    elif args.registry_command == "suggest-header":
        # Run suggest_header.py
        cmd = ["uv", "run", "python", str(project_root / "AgentQMS" / "tools" / "suggest_header.py"), args.file]

        if args.apply:
            cmd.append("--apply")
        if args.force:
            cmd.append("--force")

        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode

    elif args.registry_command == "validate":
        # Run validation using sync_registry.py in dry-run mode
        cmd = ["uv", "run", "python", str(project_root / "AgentQMS" / "tools" / "sync_registry.py"), "--dry-run"]

        if args.strict:
            cmd.append("--strict")

        result = subprocess.run(cmd, cwd=project_root)
        return result.returncode

    else:
        print("Error: Unknown registry command")
        return 1


def run_check_infra_command(args):
    """Execute check-infra subcommand."""
    from AgentQMS.tools.compliance.preflight_check import PreflightCheck

    checker = PreflightCheck()
    success = checker.print_report()
    return 0 if success else 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AgentQMS Unified CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--version", action="version", version="AgentQMS CLI v1.1.0 (ADS v2.0)")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup all subcommands
    setup_artifact_parser(subparsers)
    setup_validate_parser(subparsers)
    setup_monitor_parser(subparsers)
    setup_feedback_parser(subparsers)
    setup_quality_parser(subparsers)
    setup_generate_config_parser(subparsers)
    setup_registry_parser(subparsers)
    setup_check_infra_parser(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    # Route to appropriate command handler
    try:
        if args.command == "artifact":
            return run_artifact_command(args)
        elif args.command == "validate":
            return run_validate_command(args)
        elif args.command == "monitor":
            return run_monitor_command(args)
        elif args.command == "feedback":
            return run_feedback_command(args)
        elif args.command == "quality":
            return run_quality_command(args)
        elif args.command == "generate-config":
            return run_generate_config_command(args)
        elif args.command == "registry":
            return run_registry_command(args)
        elif args.command == "check-infra":
            return run_check_infra_command(args)
        else:
            print(f"Error: Unknown command '{args.command}'")
            return 1

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
