import os
import shutil
from pathlib import Path

def run_maintenance_audit(dry_run: bool = False):
    """
    Scans the project for stale artifacts and moves them to .archive/
    """
    project_root = Path.cwd()
    archive_dir = project_root / "AgentQMS" / ".archive"
    
    if not dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)

    # Patterns to archive
    patterns = ["*-walkthrough.md", "*_deprecated.md"]
    
    files_to_archive = []
    
    # Scan AgentQMS directory
    agent_qms_dir = project_root / "AgentQMS"
    if agent_qms_dir.exists():
        for pattern in patterns:
            files_to_archive.extend(list(agent_qms_dir.glob(pattern)))

    if not files_to_archive:
        print("No stale files found.")
        return

    print("--- Janitor Audit ---")
    for file in files_to_archive:
        target = archive_dir / file.name
        if dry_run:
            print(f"[DRY RUN] Would move: {file} -> {target}")
        else:
            print(f"Moving: {file} -> {target}")
            shutil.move(str(file), str(target))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AgentQMS Janitor")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without moving files")
    args = parser.parse_args()
    
    run_maintenance_audit(dry_run=args.dry_run)
