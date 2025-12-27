import argparse
import os
import sys

from etk.core import ExperimentTracker
from etk.factory import ExperimentFactory
from etk.reconciler import ExperimentReconciler

ETK_VERSION = "1.0.0"
EDS_VERSION = "1.0"


def main():
    parser = argparse.ArgumentParser(
        description="ETK (Experiment Tracker Kit) - CLI for EDS v1.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  etk init image_preprocessing_optimization
  etk task "Run initial baseline"
  etk create report "Performance metrics" --metrics "accuracy,f1"
  etk record path/to/artifact.md --type report
  etk status
  etk list
        """,
    )

    parser.add_argument("--version", action="version", version=f"ETK v{ETK_VERSION} | EDS v{EDS_VERSION}")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize new experiment")
    init_parser.add_argument("name", help="Experiment name (will be slugified)")
    init_parser.add_argument("--type", default="generic", help="Experiment type")
    init_parser.add_argument("--intention", "-i", default="", help="Experiment intention/goal")

    # task command
    task_parser = subparsers.add_parser("task", help="Add a task to the experiment")
    task_parser.add_argument("description", help="Task description")
    task_parser.add_argument("--experiment", "-e", help="Experiment ID (auto-detected if omitted)")

    # record command
    record_parser = subparsers.add_parser("record", help="Record an existing artifact")
    record_parser.add_argument("path", help="Path to the artifact file")
    record_parser.add_argument("--type", required=True, choices=["assessment", "report", "guide", "script", "other"], help="Artifact type")
    record_parser.add_argument("--experiment", "-e", help="Experiment ID (auto-detected if omitted)")

    # create command (Legacy wrapper / Generator)
    create_parser = subparsers.add_parser("create", help="Create and record a new artifact")
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

    # reconcile command
    reconcile_parser = subparsers.add_parser("reconcile", help="Reconcile manifest with filesystem")
    reconcile_parser.add_argument("experiment_id", nargs="?", help="Experiment ID (auto-detected if omitted)")
    reconcile_parser.add_argument("--all", action="store_true", help="Reconcile all experiments")

    # list command
    subparsers.add_parser("list", help="List all experiments")

    # sync command
    sync_parser = subparsers.add_parser("sync", help="Sync artifacts to database")
    sync_parser.add_argument("experiment_id", nargs="?", help="Experiment ID (auto-detected if omitted)")
    sync_parser.add_argument("--all", action="store_true", help="Sync all experiments")

    # prune command
    prune_parser = subparsers.add_parser("prune", help="Remove missing experiments from database")
    prune_parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting")

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
        # Initialize Factory for state changes
        # We assume CWD as base for now, or we could find a common root
        factory = ExperimentFactory(base_dir=os.getcwd() if os.path.basename(os.getcwd()) == "experiments" else "experiments")

        # Initialize Tracker for read operations
        tracker = ExperimentTracker()

        if args.command == "init":
            manifest = factory.init_experiment(args.name, args.type, args.intention)
            print(f"✅ Initialized experiment: {manifest.experiment_id}")
            print(f"📂 Location: {factory._get_experiment_dir(manifest.experiment_id)}")

        elif args.command == "task":
            # Auto-detect experiment if not provided
            exp_id = args.experiment or tracker.get_current_experiment()
            if not exp_id:
                print("❌ Error: No experiment specified and none could be detected.")
                sys.exit(1)

            task = factory.add_task(exp_id, args.description)
            print(f"✅ Added task: [{task.id}] {task.description}")

        elif args.command == "record":
            exp_id = args.experiment or tracker.get_current_experiment()
            if not exp_id:
                print("❌ Error: No experiment specified and none could be detected.")
                sys.exit(1)

            artifact = factory.record_artifact(exp_id, args.path, args.type)
            print(f"✅ Recorded artifact: {artifact.path}")

        elif args.command == "create":
            # Hybrid: Use Tracker to generate content (template), then Factory to record
            # Or just update Tracker to use Factory.
            # For now, we will use tracker.create_artifact which writes the file.
            # Then we might need to record it if the Tracker doesn't use the manifest yet?
            # Wait, Tracker.create_artifact writes to .metadata/.
            # The Factory writes to manifest.json.
            # We need to bridge this.

            kwargs = {"tags": [t.strip() for t in args.tags.split(",")] if args.tags else []}
            if args.type == "assessment":
                if args.phase:
                    kwargs["phase"] = args.phase
                if args.priority:
                    kwargs["priority"] = args.priority
            elif args.type == "report":
                if args.metrics:
                    kwargs["metrics"] = [m.strip() for m in args.metrics.split(",")]
                if args.baseline:
                    kwargs["baseline"] = args.baseline

            # Use Tracker to generate the file
            path = tracker.create_artifact(artifact_type=args.type, title=args.title, experiment_id=args.experiment, **kwargs)

            # Now Record it in Manifest via Factory
            # We need to extract the experiment ID used by tracker
            exp_id = args.experiment or tracker.get_current_experiment()

            # Make path relative to experiment dir for cleaner manifest
            try:
                exp_dir = factory._get_experiment_dir(exp_id)
                rel_path = path.relative_to(exp_dir)
                factory.record_artifact(exp_id, str(rel_path), args.type)
                print(f"✅ Recorded in manifest: {rel_path}")
            except Exception as e:
                print(f"⚠️  Warning: File created but failed to record in manifest: {e}")

        elif args.command == "status":
            # Use Tracker for status summary
            status = tracker.get_status(args.experiment_id)
            print(f"\\n📊 Experiment Status: {status['experiment_id']}")
            print(f"📂 Location: {status['path']}")
            print(f"📦 Total Artifacts: {status['total_artifacts']}")
            # We could also read the manifest via Factory to show tasks?
            try:
                manifest = factory._load_manifest(status["experiment_id"])
                print(f"📋 Tasks: {len(manifest.tasks)}")
                pending = sum(1 for t in manifest.tasks if t.status != "done")
                print(f"   Now: {pending} pending")
            except:
                pass

        elif args.command == "validate":
            is_valid, output = tracker.validate(args.experiment_id, args.all)
            for line in output:
                if line.strip():
                    print(line)
            if is_valid:
                print("\\n✅ Validation passed")
                sys.exit(0)
            else:
                print("\n❌ Validation failed")
                sys.exit(1)

        elif args.command == "reconcile":
            if args.all:
                experiments = []
                # Scan filesystem directly to bypass empty DB
                for item in tracker.experiments_dir.iterdir():
                    if item.is_dir() and not item.name.startswith("."):
                         experiments.append(item.name)

                if not experiments:
                    print("No experiments found in filesystem.")
                    sys.exit(0)

                print(f"🔄 Reconciling {len(experiments)} experiments...")
                success_count = 0
                for exp_id in experiments:
                    try:
                        exp_dir = tracker.experiments_dir / exp_id
                        reconciler = ExperimentReconciler(exp_dir)
                        result = reconciler.reconcile()
                        print(f"✅ [{exp_id}] Synced {result['artifacts_count']} artifacts")
                        success_count += 1
                    except Exception as e:
                        print(f"❌ [{exp_id}] Failed: {e}")

                print(f"\nCompleted: {success_count}/{len(experiments)} reconciled.")
                sys.exit(0 if success_count == len(experiments) else 1)

            exp_id = args.experiment_id or tracker.get_current_experiment()
            if not exp_id:
                print("❌ Error: No experiment specified and none could be detected.")
                sys.exit(1)

            try:
                exp_dir = factory._get_experiment_dir(exp_id)
                reconciler = ExperimentReconciler(exp_dir)
                result = reconciler.reconcile()
                print(f"✅ Reconciliation complete for {exp_id}")
                print(f"   artifacts found: {result['artifacts_count']}")
                print(f"   last_reconciled: {result['last_reconciled']}")
            except Exception as e:
                print(f"❌ Error reconciling experiment: {e}")
                sys.exit(1)

        elif args.command == "list":
            experiments = tracker.list_experiments()
            if not experiments:
                print("No experiments found.")
                sys.exit(0)
            print(f"\\n📊 Total Experiments: {len(experiments)}\\n")
            for exp in experiments:
                print(f"📦 {exp['experiment_id']}")

        elif args.command == "sync":
            print("🔄 Syncing artifacts to database...")
            stats = tracker.sync_to_database(args.experiment_id, args.all)
            print("\\n✅ Sync complete:")
            print(f"   ✓ Synced: {stats['synced']}")
            if stats["skipped"] > 0:
                print(f"   ⊘ Skipped: {stats['skipped']}")
            if stats["failed"] > 0:
                print(f"   ✗ Failed: {stats['failed']}")

        elif args.command == "prune":
            print("🔄 Pruning missing experiments from database...")
            stats = tracker.prune_missing(dry_run=args.dry_run)
            if args.dry_run:
                print(f"\\n🔍 Dry run complete. Found {stats['pruned']} stale experiments.")
            else:
                print(f"\\n✅ Prune complete. Removed {stats['pruned']} stale experiments.")
                print(f"   Active experiments: {stats['active']}")

        elif args.command == "query":
            results = tracker.query_artifacts(args.query)
            if not results:
                print(f"\\n❌ No results found for: {args.query}")
                sys.exit(0)
            print(f"\\n🔍 Found {len(results)} results for: {args.query}\\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['type']}] {result['title']}")
                print(f"   File: {result['file_path']}")
                print()

        elif args.command == "analytics":
            analytics = tracker.get_analytics()
            print("\\n📊 Analytics Dashboard")
            print(f"Total Experiments: {analytics['experiments']['total']}")
            print(f"Total Artifacts: {analytics['total_artifacts']}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}", file=sys.stderr)
        # import traceback; traceback.print_exc() # Debugging
        sys.exit(1)


if __name__ == "__main__":
    main()
