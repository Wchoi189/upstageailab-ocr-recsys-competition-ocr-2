#!/usr/bin/env python3
"""
Bulk migration of all experiments with state.yml to new state management.

Usage:
    python migrate_all_experiments.py [--dry-run]

Options:
    --dry-run: Show what would be migrated without making changes
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime


def find_experiments_to_migrate(tracker_root: Path):
    """Find all experiments with state.yml but no .state file."""
    experiments_dir = tracker_root / "experiments"

    if not experiments_dir.exists():
        return []

    to_migrate = []
    skipped = []

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue

        experiment_id = exp_dir.name
        state_yml = exp_dir / "state.yml"
        state_file = exp_dir / ".state"

        if state_yml.exists() and not state_file.exists():
            to_migrate.append(experiment_id)
        elif state_file.exists():
            skipped.append(experiment_id)

    return to_migrate, skipped


def main():
    dry_run = "--dry-run" in sys.argv

    # Detect tracker root
    current = Path.cwd()
    tracker_root = None
    while current != current.parent:
        if (current / "experiment-tracker").exists():
            tracker_root = current / "experiment-tracker"
            break
        if (current / "experiments").exists():
            tracker_root = current
            break
        current = current.parent

    if not tracker_root:
        print("ERROR: Could not detect experiment-tracker root")
        sys.exit(1)

    print(f"Tracker root: {tracker_root}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE MIGRATION'}")
    print()

    # Find experiments
    to_migrate, skipped = find_experiments_to_migrate(tracker_root)

    print(f"Found experiments:")
    print(f"  To migrate: {len(to_migrate)}")
    print(f"  Already migrated (skipped): {len(skipped)}")
    print()

    if not to_migrate:
        print("✅ No experiments need migration")
        return

    # Check if active experiment should be skipped
    current_file = tracker_root / "experiments" / ".current"
    active_experiment = None
    if current_file.exists():
        active_experiment = current_file.read_text().strip()
        if active_experiment in to_migrate:
            print(f"⚠️  ALERT: Active experiment will be migrated: {active_experiment}")
            print("    This is generally safe, but verify the experiment is not running.")
            print()

    if dry_run:
        print("DRY RUN - Would migrate:")
        for exp_id in to_migrate:
            marker = " [ACTIVE]" if exp_id == active_experiment else ""
            print(f"  - {exp_id}{marker}")
        print(f"\nTotal: {len(to_migrate)} experiments")
        return

    # Confirm migration
    print(f"About to migrate {len(to_migrate)} experiments.")
    response = input("Continue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled")
        return

    # Run migration
    migration_script = tracker_root / "scripts" / "migrate_state_yaml_to_db.py"
    if not migration_script.exists():
        print(f"ERROR: Migration script not found: {migration_script}")
        sys.exit(1)

    succeeded = []
    failed = []

    for i, exp_id in enumerate(to_migrate, 1):
        print(f"\n[{i}/{len(to_migrate)}] Migrating: {exp_id}")
        try:
            result = subprocess.run(
                [sys.executable, str(migration_script), exp_id],
                cwd=tracker_root,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print(result.stdout)
                succeeded.append(exp_id)
            else:
                print(f"  ❌ FAILED:")
                print(result.stderr)
                failed.append(exp_id)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed.append(exp_id)

    # Generate report
    report_path = tracker_root / "experiments_migrated.txt"
    with open(report_path, 'w') as f:
        f.write(f"Migration Report - {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total experiments: {len(to_migrate)}\n")
        f.write(f"Succeeded: {len(succeeded)}\n")
        f.write(f"Failed: {len(failed)}\n\n")

        if succeeded:
            f.write("Migrated successfully:\n")
            for exp_id in succeeded:
                f.write(f"  ✅ {exp_id}\n")
            f.write("\n")

        if failed:
            f.write("Migration failed:\n")
            for exp_id in failed:
                f.write(f"  ❌ {exp_id}\n")
            f.write("\n")

    # Summary
    print("\n" + "=" * 60)
    print("MIGRATION SUMMARY")
    print("=" * 60)
    print(f"Total: {len(to_migrate)}")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Failed: {len(failed)}")
    print(f"\nReport saved to: {report_path}")

    if failed:
        print("\n⚠️  Some migrations failed. Review the report for details.")
        sys.exit(1)
    else:
        print("\n✅ All migrations completed successfully")


if __name__ == "__main__":
    main()
