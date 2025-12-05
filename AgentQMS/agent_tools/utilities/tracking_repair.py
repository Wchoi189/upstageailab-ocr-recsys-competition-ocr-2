#!/usr/bin/env python3
"""
Tracking Database Repair Tool

Scans tracking.db for stale artifact paths and synchronizes them with current
locations from MASTER_INDEX.md. Detects moved/renamed artifacts and updates
database records accordingly.

Usage:
    python tracking_repair.py [--dry-run] [--db PATH] [--index PATH]
    make track-repair [DRY_RUN=1]

Examples:
    # Preview changes without applying
    python tracking_repair.py --dry-run

    # Apply repairs to tracking database
    python tracking_repair.py

    # Custom DB and index paths
    python tracking_repair.py --db custom.db --index custom_index.md
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

# Repository root detection
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
TRACKING_DB = REPO_ROOT / "data" / "ops" / "tracking.db"
MASTER_INDEX = REPO_ROOT / "docs" / "artifacts" / "MASTER_INDEX.md"
STATE_FILE = REPO_ROOT / ".agentqms" / "state" / "tracking_repair_state.json"


class TrackingRepair:
    """Repair tool for tracking database artifact path synchronization."""

    def __init__(
        self,
        db_path: Path = TRACKING_DB,
        index_path: Path = MASTER_INDEX,
        dry_run: bool = False
    ):
        self.db_path = db_path
        self.index_path = index_path
        self.dry_run = dry_run
        self.current_artifacts: dict[str, Path] = {}
        self.stale_paths: list[tuple[str, str, str]] = []  # (table, id, old_path)
        self.repairs: list[tuple[str, str, str, str]] = []  # (table, id, old, new)

    def run(self) -> int:
        """Execute the repair workflow."""
        print("🔧 Tracking Database Repair Tool")
        print(f"Database: {self.db_path}")
        print(f"Index: {self.index_path}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print()

        # Step 1: Load current artifact locations from MASTER_INDEX.md
        if not self._load_master_index():
            return 1

        # Step 2: Connect to tracking database
        if not self.db_path.exists():
            print(f"❌ Tracking database not found: {self.db_path}")
            print("💡 Run 'make track-init' to create the database")
            return 1

        # Step 3: Scan database for artifact references
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._scan_feature_plans(conn)
                self._scan_experiments(conn)
                self._scan_debug_sessions(conn)
        except sqlite3.Error as e:
            print(f"❌ Database error: {e}")
            return 1

        # Step 4: Report findings
        self._report_findings()

        # Step 5: Apply repairs (if not dry-run)
        if not self.dry_run and self.repairs:
            return self._apply_repairs()

        # Step 6: Save state
        self._save_state()

        return 0

    def _load_master_index(self) -> bool:
        """Load current artifact locations from MASTER_INDEX.md."""
        if not self.index_path.exists():
            print(f"❌ Master index not found: {self.index_path}")
            return False

        content = self.index_path.read_text(encoding="utf-8")

        # Parse markdown links: [title](path)
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^\)]+)\)')

        for match in link_pattern.finditer(content):
            title, path_str = match.groups()

            # Resolve relative paths
            if path_str.startswith("./"):
                path_str = path_str[2:]

            artifact_path = REPO_ROOT / "docs" / "artifacts" / path_str

            # Extract artifact key from filename
            filename = Path(path_str).name
            key = self._extract_artifact_key(filename)

            if key:
                self.current_artifacts[key] = artifact_path

        print(f"✅ Loaded {len(self.current_artifacts)} artifacts from index")
        return True

    def _extract_artifact_key(self, filename: str) -> str | None:
        """
        Extract artifact key from filename.

        Examples:
            2025-12-06_0112_implementation_plan_agentqms.md -> agentqms
            2025-12-02_2313_BUG_001_overlay.md -> overlay
        """
        # Remove timestamp prefix (YYYY-MM-DD_HHMM_)
        pattern = r'^\d{4}-\d{2}-\d{2}_\d{4}_'
        cleaned = re.sub(pattern, '', filename)

        # Remove artifact type prefix
        type_prefixes = [
            'implementation_plan_', 'assessment-', 'audit-',
            'design-', 'research-', 'BUG_', 'SESSION_'
        ]
        for prefix in type_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                break

        # Remove extension
        cleaned = cleaned.rsplit('.', 1)[0]

        return cleaned if cleaned else None

    def _scan_feature_plans(self, conn: sqlite3.Connection) -> None:
        """Scan feature_plans table for artifact_path references."""
        cursor = conn.cursor()

        # Check if table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feature_plans'"
        )
        if not cursor.fetchone():
            print("⚠️  Table 'feature_plans' not found in database")
            return

        # Check if artifact_path column exists
        cursor.execute("PRAGMA table_info(feature_plans)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'artifact_path' not in columns:
            print("ℹ️  Column 'artifact_path' not present in feature_plans")
            return

        # Scan for artifact paths
        cursor.execute("SELECT id, key, artifact_path FROM feature_plans WHERE artifact_path IS NOT NULL")

        for row in cursor.fetchall():
            plan_id, key, artifact_path = row
            self._check_path('feature_plans', plan_id, key, artifact_path)

    def _scan_experiments(self, conn: sqlite3.Connection) -> None:
        """Scan experiments table for artifact_path references."""
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'"
        )
        if not cursor.fetchone():
            print("⚠️  Table 'experiments' not found in database")
            return

        cursor.execute("PRAGMA table_info(experiments)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'artifact_path' not in columns:
            print("ℹ️  Column 'artifact_path' not present in experiments")
            return

        cursor.execute("SELECT id, key, artifact_path FROM experiments WHERE artifact_path IS NOT NULL")

        for row in cursor.fetchall():
            exp_id, key, artifact_path = row
            self._check_path('experiments', exp_id, key, artifact_path)

    def _scan_debug_sessions(self, conn: sqlite3.Connection) -> None:
        """Scan debug_sessions table for artifact_path references."""
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='debug_sessions'"
        )
        if not cursor.fetchone():
            print("⚠️  Table 'debug_sessions' not found in database")
            return

        cursor.execute("PRAGMA table_info(debug_sessions)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'artifact_path' not in columns:
            print("ℹ️  Column 'artifact_path' not present in debug_sessions")
            return

        cursor.execute("SELECT id, key, artifact_path FROM debug_sessions WHERE artifact_path IS NOT NULL")

        for row in cursor.fetchall():
            session_id, key, artifact_path = row
            self._check_path('debug_sessions', session_id, key, artifact_path)

    def _check_path(
        self,
        table: str,
        record_id: int,
        key: str,
        artifact_path: str
    ) -> None:
        """Check if artifact path is stale and needs repair."""
        path = Path(artifact_path)

        # Check if path exists
        if path.exists():
            return  # Path is valid

        # Path is stale - look for current location
        if key in self.current_artifacts:
            current_path = self.current_artifacts[key]
            if current_path.exists():
                # Found the artifact at a new location
                self.stale_paths.append((table, record_id, artifact_path))
                self.repairs.append((table, record_id, artifact_path, str(current_path)))
            else:
                # Artifact exists in index but not on filesystem
                print(f"⚠️  Artifact in index but missing: {key} -> {current_path}")
        else:
            # Artifact not found in current index
            self.stale_paths.append((table, record_id, artifact_path))
            print(f"⚠️  Artifact not in index: {key} (from {table} id={record_id})")

    def _report_findings(self) -> None:
        """Report scan results."""
        print()
        print("=" * 60)
        print("SCAN RESULTS")
        print("=" * 60)

        if not self.stale_paths:
            print("✅ No stale paths found - tracking database is synchronized")
            return

        print(f"❌ Found {len(self.stale_paths)} stale path(s)")
        print()

        for table, record_id, old_path in self.stale_paths:
            print(f"  📁 {table} (id={record_id})")
            print(f"     Old: {old_path}")

            # Find matching repair
            for r_table, r_id, r_old, r_new in self.repairs:
                if r_table == table and r_id == record_id:
                    print(f"     New: {r_new}")
                    break
            else:
                print("     New: [NOT FOUND IN INDEX]")
            print()

        if self.repairs:
            print(f"🔧 {len(self.repairs)} repair(s) available")

        if self.dry_run:
            print("ℹ️  DRY RUN mode - no changes will be applied")

    def _apply_repairs(self) -> int:
        """Apply repairs to tracking database."""
        print()
        print("=" * 60)
        print("APPLYING REPAIRS")
        print("=" * 60)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for table, record_id, old_path, new_path in self.repairs:
                    print(f"🔧 Updating {table} id={record_id}")
                    cursor.execute(
                        f"UPDATE {table} SET artifact_path = ? WHERE id = ?",
                        (new_path, record_id)
                    )

                conn.commit()
                print()
                print(f"✅ Successfully applied {len(self.repairs)} repair(s)")

        except sqlite3.Error as e:
            print(f"❌ Failed to apply repairs: {e}")
            return 1

        return 0

    def _save_state(self) -> None:
        """Save repair state for audit trail."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "stale_paths_count": len(self.stale_paths),
            "repairs_applied": len(self.repairs) if not self.dry_run else 0,
            "dry_run": self.dry_run,
            "repairs": [
                {
                    "table": table,
                    "id": record_id,
                    "old_path": old_path,
                    "new_path": new_path
                }
                for table, record_id, old_path, new_path in self.repairs
            ]
        }

        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"💾 State saved to {STATE_FILE}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Repair tracking database artifact path references",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=TRACKING_DB,
        help=f"Path to tracking database (default: {TRACKING_DB})"
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=MASTER_INDEX,
        help=f"Path to master index (default: {MASTER_INDEX})"
    )

    args = parser.parse_args()

    repair = TrackingRepair(
        db_path=args.db,
        index_path=args.index,
        dry_run=args.dry_run
    )

    return repair.run()


if __name__ == "__main__":
    sys.exit(main())
