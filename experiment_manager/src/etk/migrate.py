from pathlib import Path

LEGACY_SCRIPTS = [
    "add-task.py",
    "append-run.py",
    "batch_import.py",
    "complete-task.py",
    "export-experiment.py",
    "generate-assessment.py",
    "generate-feedback.py",
    "generate-incident-report.py",
    "log-insight.py",
    "migrate_all_experiments.py",
    "migrate_state_to_yaml.py",
    "migrate_state_yaml_to_db.py",
    "migrate_to_v2.py",
    "record-artifact.py",
    "record-decision.py",
    "reorganize_experiment.py",
    "resume-experiment.py",
    "safe_state_manager.py",
    "start-experiment.py",
    "stash-incomplete.py",
    "workflow.py",
]


def migrate():
    base_dir = Path(__file__).resolve().parent.parent.parent
    scripts_dir = base_dir / "scripts"

    print(f"Checking for legacy scripts in: {scripts_dir}")

    if not scripts_dir.exists():
        print("Scripts directory not found.")
        return

    # 1. Check for State Files (Migration Step)
    # Since experiments directory was verified empty, we skip complex state migration.
    # However, for completeness, we'd log if we found any.
    experiments_dir = base_dir / "experiments"
    if experiments_dir.exists() and any(experiments_dir.iterdir()):
        print(
            "WARNING: Experiments directory is not empty. Please verify migration of data manually as this script is in 'Kill Only' mode."
        )
    else:
        print("Experiments directory is empty or does not exist. Skipping data migration.")

    # 2. Delete Legacy Scripts
    for script_name in LEGACY_SCRIPTS:
        script_path = scripts_dir / script_name
        if script_path.exists():
            print(f"Removing legacy script: {script_name}")
            # Uncomment for actual deletion
            script_path.unlink()
        else:
            print(f"Script not found (already gone?): {script_name}")

    print("\nMigration/Cleanup Complete.")


if __name__ == "__main__":
    migrate()
