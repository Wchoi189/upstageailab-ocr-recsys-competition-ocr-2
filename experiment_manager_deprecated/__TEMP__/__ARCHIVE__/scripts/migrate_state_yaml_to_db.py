#!/usr/bin/env python3
"""
Migrate state.yml to .state file + database tables.

Usage:
    python migrate_state_yaml_to_db.py <experiment_id>

Example:
    python migrate_state_yaml_to_db.py 20251217_024343_image_enhancements_implementation
"""

import shutil
import sqlite3
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from experiment_manager.utils.state_file import create_state_file


def migrate_experiment(experiment_id: str, tracker_root: Path):
    """Migrate single experiment from state.yml to new format."""
    exp_path = tracker_root / "experiments" / experiment_id
    state_yml = exp_path / "state.yml"

    if not state_yml.exists():
        print(f"Skip: No state.yml found for {experiment_id}")
        return False

    # Skip if .state already exists
    if (exp_path / ".state").exists():
        print(f"Skip: .state already exists for {experiment_id}")
        return False

    print(f"Migrating: {experiment_id}")

    # Load YAML state
    with open(state_yml) as f:
        state_data = yaml.safe_load(f)

    if not state_data:
        print("  Warning: Empty state.yml, skipping")
        return False

    # Extract core state fields
    experiment_id_from_yaml = state_data.get("experiment_id", experiment_id)
    status = state_data.get("status", "active")
    current_phase = state_data.get("phase", "planning")
    checkpoint_path = state_data.get("checkpoint", {}).get("path") if isinstance(state_data.get("checkpoint"), dict) else None

    # Create .state file
    create_state_file(exp_path, experiment_id_from_yaml, checkpoint_path=checkpoint_path)
    print(f"  Created .state file ({(exp_path / '.state').stat().st_size} bytes)")

    # Get database connection
    db_path = tracker_root.parent / "data" / "ops" / "tracking.db"
    if not db_path.exists():
        print("  Warning: Database not found, skipping DB migration")
        return True

    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now(UTC).isoformat()

        # Populate experiment_state table
        conn.execute(
            """
            INSERT OR REPLACE INTO experiment_state (
                experiment_id, current_task_id, current_phase, status,
                created_at, updated_at, checkpoint_path
            ) VALUES (?, NULL, ?, ?, ?, ?, ?)
        """,
            (experiment_id, current_phase, status, now, now, checkpoint_path),
        )

        # Migrate tasks array
        tasks = state_data.get("tasks", [])
        tasks_migrated = 0
        for task in tasks:
            if isinstance(task, dict):
                task_id = f"{task.get('id', task.get('title', 'task').lower().replace(' ', '_'))}_{uuid.uuid4().hex[:8]}"
                title = task.get("title", task.get("name", "Untitled"))
                description = task.get("notes", task.get("description", ""))
                status_field = task.get("status", "backlog")
                priority = task.get("priority", "medium")
                completed_at = task.get("completed_at")

                conn.execute(
                    """
                    INSERT INTO experiment_tasks (
                        task_id, experiment_id, title, description,
                        status, priority, created_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (task_id, experiment_id, title, description, status_field, priority, now, completed_at),
                )
                tasks_migrated += 1

        if tasks_migrated > 0:
            print(f"  Migrated {tasks_migrated} tasks to database")

        # Migrate decisions array
        decisions = state_data.get("decisions", [])
        decisions_migrated = 0
        for decision in decisions:
            if isinstance(decision, dict):
                decision_id = f"dec_{uuid.uuid4().hex[:12]}"
                date = decision.get("date", datetime.now().date().isoformat())
                decision_text = decision.get("decision", decision.get("text", "N/A"))
                rationale = decision.get("rationale", decision.get("reasoning", "N/A"))
                impact = decision.get("impact", "")

                conn.execute(
                    """
                    INSERT INTO experiment_decisions (
                        decision_id, experiment_id, date, decision, rationale, impact, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (decision_id, experiment_id, date, decision_text, rationale, impact, now),
                )
                decisions_migrated += 1

        if decisions_migrated > 0:
            print(f"  Migrated {decisions_migrated} decisions to database")

        # Migrate insights array
        insights = state_data.get("insights", [])
        insights_migrated = 0
        for insight in insights:
            if isinstance(insight, dict):
                insight_id = f"ins_{uuid.uuid4().hex[:12]}"
                date = insight.get("date", datetime.now().date().isoformat())
                insight_text = insight.get("insight", insight.get("text", "N/A"))
                impact = insight.get("impact", "N/A")
                category = insight.get("category", "observation")

                conn.execute(
                    """
                    INSERT INTO experiment_insights (
                        insight_id, experiment_id, date, insight, impact, category, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (insight_id, experiment_id, date, insight_text, impact, category, now),
                )
                insights_migrated += 1

        if insights_migrated > 0:
            print(f"  Migrated {insights_migrated} insights to database")

        conn.commit()

    finally:
        conn.close()

    # Backup original state.yml
    archive_dir = exp_path / ".metadata" / "archive"
    archive_dir.mkdir(exist_ok=True)
    backup_path = archive_dir / f"state.yml.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(state_yml, backup_path)
    print(f"  Backed up state.yml to {backup_path.relative_to(exp_path)}")

    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python migrate_state_yaml_to_db.py <experiment_id>")
        print("\nExample:")
        print("  python migrate_state_yaml_to_db.py 20251217_024343_image_enhancements_implementation")
        sys.exit(1)

    experiment_id = sys.argv[1]

    # Detect tracker root
    current = Path.cwd()
    tracker_root = None
    while current != current.parent:
        if (current / "experiment_manager").exists():
            tracker_root = current / "experiment_manager"
            break
        if (current / "experiments").exists():
            tracker_root = current
            break
        current = current.parent

    if not tracker_root:
        print("ERROR: Could not detect experiment_manager root")
        sys.exit(1)

    print(f"Tracker root: {tracker_root}")

    success = migrate_experiment(experiment_id, tracker_root)

    if success:
        print(f"\n✅ Migration complete for {experiment_id}")
        print("\n⚠️  IMPORTANT:")
        print("  - .state file created (100-byte core state)")
        print("  - Tasks/decisions/insights moved to database")
        print("  - Original state.yml backed up in .metadata/archive/")
        print("  - You can now safely delete state.yml if desired")
    else:
        print(f"\n❌ Migration failed or skipped for {experiment_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
