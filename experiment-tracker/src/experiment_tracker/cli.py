import argparse
import sys
from pathlib import Path
from experiment_tracker.core import ExperimentTracker

ETK_VERSION = "1.0.0"
EDS_VERSION = "1.0"

def main():
    parser = argparse.ArgumentParser(
        description="ETK (Experiment Tracker Kit) - CLI for EDS v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  etk init image_preprocessing_optimization
  etk create assessment "Initial baseline evaluation"
  etk create report "Performance metrics" --metrics "accuracy,f1"
  etk status
  etk validate --all
  etk list
        """,
    )

    parser.add_argument("--version", action="version", version=f"ETK v{ETK_VERSION} | EDS v{EDS_VERSION}")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize new experiment")
    init_parser.add_argument("name", help="Experiment name (will be slugified)")
    init_parser.add_argument("--description", "-d", default="", help="Experiment description")
    init_parser.add_argument("--tags", "-t", default="", help="Comma-separated tags")

    # create command
    create_parser = subparsers.add_parser("create", help="Create artifact")
    create_parser.add_argument("type", choices=["assessment", "report", "guide", "script"], help="Artifact type")
    create_parser.add_argument("title", help="Artifact title")
    create_parser.add_argument("--experiment", "-e", help="Experiment ID (auto-detected if omitted)")
    create_parser.add_argument("--phase", choices=["planning", "execution", "analysis", "complete"], help="Assessment phase")
    create_parser.add_argument("--priority", choices=["low", "medium", "high", "critical"], help="Assessment priority")
    create_parser.add_argument("--metrics", help="Comma-separated metrics (for reports)")
    create_parser.add_argument("--baseline", help="Baseline identifier (for reports)")
    create_parser.add_argument("--tags", "-t", help="Comma-separated tags")

    # status command
    status_parser = subparsers.add_parser("status", help="Show experiment status")
    status_parser.add_argument("experiment_id", nargs="?", help="Experiment ID (auto-detected if omitted)")

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate experiment compliance")
    validate_parser.add_argument("experiment_id", nargs="?", help="Experiment ID (auto-detected if omitted)")
    validate_parser.add_argument("--all", action="store_true", help="Validate all experiments")

    # list command
    subparsers.add_parser("list", help="List all experiments")

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync artifacts to database")
    sync_parser.add_argument("experiment_id", nargs="?", help="Experiment ID (auto-detected if omitted)")
    sync_parser.add_argument("--all", action="store_true", help="Sync all experiments")

    # query command
    query_parser = subparsers.add_parser("query", help="Search artifacts (FTS5)")
    query_parser.add_argument("query", help="Search query")

    # analytics command
    subparsers.add_parser("analytics", help="Show analytics dashboard")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        tracker = ExperimentTracker()

        if args.command == "init":
            tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
            tracker.init_experiment(args.name, args.description, tags)

        elif args.command == "create":
            kwargs = {"tags": [t.strip() for t in args.tags.split(",")] if args.tags else []}
            if args.type == "assessment":
                if args.phase: kwargs["phase"] = args.phase
                if args.priority: kwargs["priority"] = args.priority
            elif args.type == "report":
                if args.metrics: kwargs["metrics"] = [m.strip() for m in args.metrics.split(",")]
                if args.baseline: kwargs["baseline"] = args.baseline
            tracker.create_artifact(artifact_type=args.type, title=args.title, experiment_id=args.experiment, **kwargs)

        elif args.command == "status":
            status = tracker.get_status(args.experiment_id)
            print(f"\n📊 Experiment Status: {status['experiment_id']}")
            print(f"📂 Location: {status['path']}")
            print("\n📋 Artifacts:")
            for artifact_type, count in status["artifacts"].items():
                icon = "✅" if count > 0 else "⚪"
                print(f"   {icon} {artifact_type}: {count}")
            print(f"\n📦 Total: {status['total_artifacts']} artifacts")

        elif args.command == "validate":
            is_valid, output = tracker.validate(args.experiment_id, args.all)
            for line in output:
                if line.strip(): print(line)
            if is_valid:
                print("\n✅ Validation passed")
                sys.exit(0)
            else:
                print("\n❌ Validation failed")
                sys.exit(1)

        elif args.command == "list":
            experiments = tracker.list_experiments()
            if not experiments:
                print("No experiments found.")
                sys.exit(0)
            print(f"\n📊 Total Experiments: {len(experiments)}\n")
            for exp in experiments:
                total = exp["total_artifacts"]
                icon = "📦" if total > 0 else "📭"
                print(f"{icon} {exp['experiment_id']}")
                print(f"   Artifacts: {total} (A:{exp['artifacts']['assessments']} R:{exp['artifacts']['reports']} G:{exp['artifacts']['guides']} S:{exp['artifacts']['scripts']})")
                print()

        elif args.command == "sync":
            print("🔄 Syncing artifacts to database...")
            stats = tracker.sync_to_database(args.experiment_id, args.all)
            print("\n✅ Sync complete:")
            print(f"   ✓ Synced: {stats['synced']}")
            if stats["skipped"] > 0: print(f"   ⊘ Skipped: {stats['skipped']}")
            if stats["failed"] > 0: print(f"   ✗ Failed: {stats['failed']}")

        elif args.command == "query":
            results = tracker.query_artifacts(args.query)
            if not results:
                print(f"\n❌ No results found for: {args.query}")
                sys.exit(0)
            print(f"\n🔍 Found {len(results)} results for: {args.query}\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['type']}] {result['title']}")
                print(f"   Experiment: {result['experiment_id']}")
                print(f"   File: {result['file_path']}")
                print(f"   Snippet: {result['snippet']}")
                print()

        elif args.command == "analytics":
            analytics = tracker.get_analytics()
            print("\n📊 Experiment Tracker Analytics Dashboard\n")
            exp = analytics["experiments"]
            print("🧪 Experiments")
            print(f"   Total: {exp['total']}")
            print(f"   Active: {exp['active']}")
            print(f"   Complete: {exp['complete']}")
            print(f"   Deprecated: {exp['deprecated']}")
            print(f"\n📋 Artifacts: {analytics['total_artifacts']} total")
            for artifact_type, count in analytics["artifacts_by_type"].items():
                print(f"   {artifact_type}: {count}")
            print("\n🕒 Recent Activity (Last 5 Updates)")
            for activity in analytics["recent_activity"][:5]:
                print(f"   [{activity['type']}] {activity['title']}")
                print(f"      Experiment: {activity['experiment_name']}")
                print(f"      Updated: {activity['updated_at']}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
