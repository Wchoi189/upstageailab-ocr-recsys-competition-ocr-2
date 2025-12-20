#!/usr/bin/env python3
"""
ETK (Experiment Tracker Kit) - CLI Tool for EDS v1.0

Comprehensive command-line interface for experiment artifact management.
Ensures 100% EDS v1.0 compliance through automated artifact generation.

Usage:
    etk init <experiment_name>                 # Initialize new experiment
    etk create assessment <title>              # Create assessment artifact
    etk create report <title>                  # Create report artifact
    etk create guide <title>                   # Create guide artifact
    etk create script <title>                  # Create script artifact
    etk status [experiment_id]                 # Show experiment status
    etk validate [experiment_id]               # Validate compliance
    etk list                                   # List all experiments
    etk sync [experiment_id]                   # Sync artifacts to database
    etk query <query_string>                   # Search artifacts (FTS5)
    etk analytics                              # Show analytics dashboard
    etk version                                # Show ETK version

Examples:
    etk init image_preprocessing_optimization
    etk create assessment "Initial baseline evaluation"
    etk create report "Performance metrics analysis" --metrics "accuracy,f1,latency"
    etk status 20251217_024343_image_enhancements
    etk validate --all
    etk sync --all
    etk query "performance optimization"
    etk analytics
"""

import argparse
import json
import re
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

# Import state file utilities
sys.path.insert(0, str(Path(__file__).parent / "src"))
from experiment_tracker.utils.state_file import create_state_file, update_state

# Define KST timezone (UTC+9)
KST = timezone(timedelta(hours=9))

ETK_VERSION = "1.0.0"
EDS_VERSION = "1.0"


class ExperimentTracker:
    """Main experiment tracker interface."""

    def __init__(self, tracker_root: Path | None = None):
        if tracker_root:
            self.tracker_root = tracker_root
        else:
            # Auto-detect tracker root
            current = Path.cwd()
            while current != current.parent:
                if (current / "experiment-tracker").exists():
                    self.tracker_root = current / "experiment-tracker"
                    break
                if (current / ".ai-instructions").exists():
                    self.tracker_root = current
                    break
                current = current.parent
            else:
                self.tracker_root = Path.cwd()

        self.experiments_dir = self.tracker_root / "experiments"
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        # Database path
        self.db_path = self.tracker_root.parent / "data" / "ops" / "tracking.db"

        # Load templates
        self.templates_path = self.tracker_root / ".ai-instructions" / "tier2-framework" / "artifact-catalog.yaml"

    def get_current_experiment(self) -> str | None:
        """Get current experiment ID from CWD or config."""
        cwd = Path.cwd()

        # Check if inside an experiment directory
        if "experiments" in cwd.parts:
            idx = cwd.parts.index("experiments")
            if len(cwd.parts) > idx + 1:
                return cwd.parts[idx + 1]

        return None

    def init_experiment(self, name: str, description: str = "", tags: list[str] = None) -> str:
        """
        Initialize new experiment with proper structure.

        Returns:
            experiment_id (str): YYYYMMDD_HHMMSS_{name}
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = name.lower().replace(" ", "_").replace("-", "_")
        experiment_id = f"{timestamp}_{slug}"

        exp_path = self.experiments_dir / experiment_id

        if exp_path.exists():
            raise ValueError(f"Experiment already exists: {experiment_id}")

        # Create directory structure
        exp_path.mkdir(parents=True)
        (exp_path / ".metadata").mkdir()
        (exp_path / ".metadata" / "assessments").mkdir()
        (exp_path / ".metadata" / "reports").mkdir()
        (exp_path / ".metadata" / "guides").mkdir()
        (exp_path / ".metadata" / "plans").mkdir()

        # NEW: Progress-tracker integration (status hierarchy)
        (exp_path / ".metadata" / "00-status").mkdir()
        (exp_path / ".metadata" / "01-work-logs").mkdir()
        (exp_path / ".metadata" / "02-tasks").mkdir()

        # Create experiment manifest
        manifest = self._generate_manifest(experiment_id, name, description, tags or [])
        manifest_path = exp_path / "README.md"
        manifest_path.write_text(manifest, encoding="utf-8")

        # NEW: Create .state file (100-byte core state)
        create_state_file(exp_path, experiment_id)

        # NEW: Populate experiment_state table in database
        self._init_experiment_in_db(experiment_id, name, description, tags or [])

        print(f"✅ Initialized experiment: {experiment_id}")
        print(f"📂 Location: {exp_path}")
        print(f"📊 State: .state file created ({Path(exp_path / '.state').stat().st_size} bytes)")
        print("\n📋 Next steps:")
        print(f"   cd {exp_path}")
        print('   etk create assessment "Initial baseline evaluation"')

        return experiment_id

    def create_artifact(self, artifact_type: str, title: str, experiment_id: str | None = None, **kwargs) -> Path:
        """
        Create new artifact with EDS v1.0 compliant frontmatter.

        Args:
            artifact_type: assessment, report, guide, script
            title: Human-readable title
            experiment_id: Target experiment (auto-detected if None)
            **kwargs: Type-specific fields

        Returns:
            Path to created artifact
        """
        # Resolve experiment
        if not experiment_id:
            experiment_id = self.get_current_experiment()
            if not experiment_id:
                raise ValueError("No experiment specified and unable to auto-detect. Use --experiment or run from experiment directory.")

        exp_path = self.experiments_dir / experiment_id
        if not exp_path.exists():
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        slug = self._generate_slug(title)
        filename = f"{timestamp}_{artifact_type}_{slug}.md"

        # Determine location
        artifact_path = exp_path / ".metadata" / f"{artifact_type}s" / filename

        if artifact_path.exists():
            raise ValueError(f"Artifact already exists: {filename}")

        # Generate content
        content = self._generate_artifact_content(artifact_type=artifact_type, title=title, experiment_id=experiment_id, **kwargs)

        # Write artifact
        artifact_path.write_text(content, encoding="utf-8")

        print(f"✅ Created {artifact_type}: {filename}")
        print(f"📂 Location: {artifact_path}")
        print(f"\n📝 Edit with: code {artifact_path}")

        return artifact_path

    def get_status(self, experiment_id: str | None = None) -> dict:
        """Get experiment status summary."""
        if not experiment_id:
            experiment_id = self.get_current_experiment()
            if not experiment_id:
                raise ValueError("No experiment specified")

        exp_path = self.experiments_dir / experiment_id
        if not exp_path.exists():
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Count artifacts by type
        metadata_path = exp_path / ".metadata"
        status = {
            "experiment_id": experiment_id,
            "path": str(exp_path),
            "artifacts": {
                "assessments": len(list((metadata_path / "assessments").glob("*.md"))) if (metadata_path / "assessments").exists() else 0,
                "reports": len(list((metadata_path / "reports").glob("*.md"))) if (metadata_path / "reports").exists() else 0,
                "guides": len(list((metadata_path / "guides").glob("*.md"))) if (metadata_path / "guides").exists() else 0,
                "scripts": len(list((metadata_path / "scripts").glob("*.md"))) if (metadata_path / "scripts").exists() else 0,
            },
            "total_artifacts": 0,
        }

        status["total_artifacts"] = sum(status["artifacts"].values())

        return status

    def validate(self, experiment_id: str | None = None, all_experiments: bool = False) -> tuple[bool, list[str]]:
        """
        Validate experiment compliance.

        Returns:
            (is_valid, errors)
        """
        # Use compliance-checker.py
        checker_path = self.tracker_root / ".ai-instructions" / "schema" / "compliance-checker.py"

        if not checker_path.exists():
            return False, ["Compliance checker not found"]

        if all_experiments:
            # Validate all experiments
            cmd = ["python3", str(checker_path), "--all"]
        else:
            if not experiment_id:
                experiment_id = self.get_current_experiment()
                if not experiment_id:
                    return False, ["No experiment specified"]

            exp_path = self.experiments_dir / experiment_id
            cmd = ["python3", str(checker_path), str(exp_path)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.tracker_root)
            return result.returncode == 0, result.stdout.split("\n")
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    def list_experiments(self) -> list[dict]:
        """List all experiments with summary info."""
        experiments = []

        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if not exp_dir.is_dir():
                continue

            status = self.get_status(exp_dir.name)
            experiments.append(status)

        return experiments

    def sync_to_database(self, experiment_id: str | None = None, sync_all: bool = False) -> dict:
        """
        Sync artifacts to database.

        Args:
            experiment_id: Specific experiment to sync (None for current)
            sync_all: Sync all experiments

        Returns:
            Statistics dict: {synced: int, failed: int, skipped: int}
        """
        stats = {"synced": 0, "failed": 0, "skipped": 0}

        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            if sync_all:
                experiments = [exp["experiment_id"] for exp in self.list_experiments()]
            elif experiment_id:
                experiments = [experiment_id]
            else:
                current = self.get_current_experiment()
                if not current:
                    raise ValueError("No experiment specified and cannot detect current experiment")
                experiments = [current]

            for exp_id in experiments:
                try:
                    self._sync_experiment(conn, exp_id, stats)
                except Exception as e:
                    print(f"⚠️  Failed to sync {exp_id}: {e}", file=sys.stderr)
                    stats["failed"] += 1

            conn.commit()

        finally:
            conn.close()

        return stats

    def _sync_experiment(self, conn: sqlite3.Connection, experiment_id: str, stats: dict):
        """Sync single experiment to database."""
        exp_path = self.experiments_dir / experiment_id

        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_id}")

        # Parse experiment info from directory
        # Extract timestamp and name from experiment_id (format: YYYYMMDD_HHMMSS_name)
        parts = experiment_id.split("_", 2)
        if len(parts) >= 3:
            timestamp_str = f"{parts[0]}_{parts[1]}"
            name = parts[2].replace("_", " ").title()
        else:
            timestamp_str = parts[0] if parts else experiment_id
            name = experiment_id

        try:
            # Parse timestamp and localize to KST
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            created_at = dt.replace(tzinfo=KST).isoformat()
        except ValueError:
            created_at = datetime.now(KST).isoformat()

        # Check if experiment exists
        cursor = conn.execute("SELECT experiment_id FROM experiments WHERE experiment_id = ?", (experiment_id,))
        exists = cursor.fetchone() is not None

        if exists:
            conn.execute("UPDATE experiments SET updated_at = ? WHERE experiment_id = ?", (datetime.now(KST).isoformat(), experiment_id))
        else:
            conn.execute(
                "INSERT INTO experiments (experiment_id, name, status, created_at, updated_at, ads_version) VALUES (?, ?, ?, ?, ?, ?)",
                (experiment_id, name, "active", created_at, datetime.now(KST).isoformat(), EDS_VERSION),
            )

        # Find all markdown files matching EDS naming pattern (YYYYMMDD_HHMM_TYPE_slug.md)
        artifact_pattern = re.compile(r"^\d{8}_\d{4}_(assessment|report|guide|script)_.*\.md$")

        # Scan experiment directory recursively
        for artifact_path in exp_path.rglob("*.md"):
            # Skip README and other non-artifact files
            if artifact_path.name.lower() in ["readme.md", "index.md"]:
                continue

            # Check if filename matches EDS pattern
            if artifact_pattern.match(artifact_path.name):
                try:
                    # Extract artifact type from filename
                    artifact_type = artifact_path.name.split("_")[2]
                    self._sync_artifact(conn, experiment_id, artifact_path, artifact_type, stats)
                except Exception as e:
                    print(f"⚠️  Failed to sync {artifact_path.name}: {e}", file=sys.stderr)
                    stats["failed"] += 1

    def _sync_artifact(self, conn: sqlite3.Connection, experiment_id: str, artifact_path: Path, artifact_type: str, stats: dict):
        """Sync single artifact to database."""
        with open(artifact_path, encoding="utf-8") as f:
            content = f.read()

        # Extract frontmatter
        frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not frontmatter_match:
            stats["skipped"] += 1
            return

        frontmatter_text = frontmatter_match.group(1)
        frontmatter = {}

        # Parse YAML-like frontmatter (simple key: value extraction)
        for line in frontmatter_text.split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                frontmatter[key] = value

        artifact_id = artifact_path.stem
        title = frontmatter.get("title", artifact_id)
        status = frontmatter.get("status", "active")
        created = frontmatter.get("created", datetime.now(KST).isoformat())
        updated = frontmatter.get("updated", datetime.now(KST).isoformat())

        # Extract main content (after frontmatter)
        main_content = content[frontmatter_match.end() :]

        # Check if artifact exists
        cursor = conn.execute("SELECT artifact_id FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        exists = cursor.fetchone() is not None

        if exists:
            conn.execute(
                "UPDATE artifacts SET title = ?, status = ?, updated_at = ? WHERE artifact_id = ?", (title, status, updated, artifact_id)
            )
        else:
            conn.execute(
                "INSERT INTO artifacts (artifact_id, experiment_id, type, title, file_path, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    artifact_id,
                    experiment_id,
                    artifact_type,
                    title,
                    str(artifact_path.relative_to(self.tracker_root)),
                    status,
                    created,
                    updated,
                ),
            )

        # Sync FTS
        conn.execute("DELETE FROM artifacts_fts WHERE artifact_id = ?", (artifact_id,))
        conn.execute(
            "INSERT INTO artifacts_fts (artifact_id, experiment_id, title, content) VALUES (?, ?, ?, ?)",
            (artifact_id, experiment_id, title, main_content),
        )

        # Sync tags
        tags_str = frontmatter.get("tags", "")
        if tags_str:
            tags = [t.strip() for t in tags_str.split(",")]
            for tag in tags:
                # Insert tag if not exists
                conn.execute("INSERT OR IGNORE INTO tags (experiment_id, tag_name) VALUES (?, ?)", (experiment_id, tag))
                # Link artifact to tag
                conn.execute("INSERT OR IGNORE INTO artifact_tags (artifact_id, tag_name) VALUES (?, ?)", (artifact_id, tag))

        # Sync type-specific metadata
        if artifact_type == "assessment":
            phase = frontmatter.get("phase")
            priority = frontmatter.get("priority")
            evidence_count = frontmatter.get("evidence_count")

            # Validate phase against CHECK constraint
            valid_phases = ["planning", "execution", "analysis", "complete"]
            if phase and phase not in valid_phases:
                phase = None

            # Validate priority against CHECK constraint
            valid_priorities = ["low", "medium", "high", "critical"]
            if priority and priority not in valid_priorities:
                priority = None

            if phase or priority or evidence_count:
                conn.execute(
                    "INSERT OR REPLACE INTO artifact_metadata (artifact_id, phase, priority, evidence_count) VALUES (?, ?, ?, ?)",
                    (artifact_id, phase, priority, int(evidence_count) if evidence_count else None),
                )

        elif artifact_type == "report":
            metrics = frontmatter.get("metrics")
            baseline = frontmatter.get("baseline")
            comparison = frontmatter.get("comparison")

            # Validate comparison against CHECK constraint
            valid_comparisons = ["baseline", "previous", "best"]
            if comparison and comparison not in valid_comparisons:
                comparison = None

            if metrics or baseline or comparison:
                conn.execute(
                    "INSERT OR REPLACE INTO artifact_metadata (artifact_id, metrics, baseline, comparison) VALUES (?, ?, ?, ?)",
                    (artifact_id, metrics, baseline, comparison),
                )

        stats["synced"] += 1

    def query_artifacts(self, query: str) -> list[dict]:
        """
        Search artifacts using FTS5.

        Args:
            query: Search query (FTS5 syntax)

        Returns:
            List of matching artifacts with snippets
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(
                """
                SELECT
                    f.artifact_id,
                    f.experiment_id,
                    f.title,
                    snippet(artifacts_fts, 3, '→ ', ' ←', '...', 32) as snippet,
                    a.type,
                    a.status,
                    a.file_path
                FROM artifacts_fts f
                JOIN artifacts a ON f.artifact_id = a.artifact_id
                WHERE artifacts_fts MATCH ?
                ORDER BY rank
                LIMIT 20
            """,
                (query,),
            )

            results = []
            for row in cursor.fetchall():
                results.append(
                    {
                        "artifact_id": row["artifact_id"],
                        "experiment_id": row["experiment_id"],
                        "title": row["title"],
                        "snippet": row["snippet"],
                        "type": row["type"],
                        "status": row["status"],
                        "file_path": row["file_path"],
                    }
                )

            return results

        finally:
            conn.close()

    def get_analytics(self) -> dict:
        """
        Generate analytics dashboard data.

        Returns:
            Analytics dict with statistics and insights
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            analytics = {}

            # Experiment counts
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as complete,
                    SUM(CASE WHEN status = 'deprecated' THEN 1 ELSE 0 END) as deprecated
                FROM experiments
            """)
            row = cursor.fetchone()
            analytics["experiments"] = {
                "total": row["total"],
                "active": row["active"],
                "complete": row["complete"],
                "deprecated": row["deprecated"],
            }

            # Artifact counts by type
            cursor = conn.execute("""
                SELECT type, COUNT(*) as count
                FROM artifacts
                GROUP BY type
                ORDER BY count DESC
            """)
            analytics["artifacts_by_type"] = {row["type"]: row["count"] for row in cursor.fetchall()}

            # Total artifacts
            cursor = conn.execute("SELECT COUNT(*) as count FROM artifacts")
            analytics["total_artifacts"] = cursor.fetchone()["count"]

            # Recent activity (last 10 artifacts)
            cursor = conn.execute("""
                SELECT
                    a.artifact_id,
                    a.title,
                    a.type,
                    a.updated_at,
                    e.name as experiment_name
                FROM artifacts a
                JOIN experiments e ON a.experiment_id = e.experiment_id
                ORDER BY a.updated_at DESC
                LIMIT 10
            """)
            analytics["recent_activity"] = [dict(row) for row in cursor.fetchall()]

            # Artifacts per experiment
            cursor = conn.execute("""
                SELECT
                    e.name,
                    e.experiment_id,
                    COUNT(a.artifact_id) as artifact_count
                FROM experiments e
                LEFT JOIN artifacts a ON e.experiment_id = a.experiment_id
                GROUP BY e.experiment_id
                ORDER BY artifact_count DESC
            """)
            analytics["artifacts_per_experiment"] = [dict(row) for row in cursor.fetchall()]

            # Tag statistics
            cursor = conn.execute("""
                SELECT
                    t.tag_name,
                    COUNT(DISTINCT at.artifact_id) as artifact_count,
                    COUNT(DISTINCT t.experiment_id) as experiment_count
                FROM tags t
                LEFT JOIN artifact_tags at ON t.tag_name = at.tag_name
                GROUP BY t.tag_name
                ORDER BY artifact_count DESC
                LIMIT 10
            """)
            analytics["popular_tags"] = [dict(row) for row in cursor.fetchall()]

            return analytics

        finally:
            conn.close()

    def _generate_slug(self, title: str) -> str:
        """Generate URL-safe slug from title."""
        slug = title.lower()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[-\s]+", "-", slug)
        slug = slug.strip("-")

        if len(slug) > 50:
            slug = slug[:50].rstrip("-")

        return slug

    def _generate_manifest(self, experiment_id: str, name: str, description: str, tags: list[str]) -> str:
        """Generate experiment manifest (README.md)."""
        now = datetime.now(KST).isoformat()

        manifest = f"""---
ads_version: "{EDS_VERSION}"
type: experiment_manifest
experiment_id: "{experiment_id}"
status: "active"
created: "{now}"
updated: "{now}"
tags: {json.dumps(tags)}
---

# Experiment: {name}

## Overview

{description or "No description provided."}

## Structure

This experiment follows EDS v1.0 (Experiment Documentation Standard).

**Artifact Locations**:
- Assessments: `.metadata/assessments/`
- Reports: `.metadata/reports/`
- Guides: `.metadata/guides/`
- Scripts: `.metadata/scripts/`
- Artifacts: `.metadata/artifacts/`

## Quick Commands

**Create artifacts**:
```bash
etk create assessment "Title here"
etk create report "Performance analysis"
etk create guide "Setup instructions"
etk create script "Automation script"
```

**Check status**:
```bash
etk status
```

**Validate compliance**:
```bash
etk validate
```

## Tags

{", ".join(tags) if tags else "No tags"}

---

*Created by ETK v{ETK_VERSION} | EDS v{EDS_VERSION}*
"""
        return manifest

    def _generate_artifact_content(self, artifact_type: str, title: str, experiment_id: str, **kwargs) -> str:
        """Generate artifact content with compliant frontmatter."""
        now = datetime.now(KST).isoformat()

        # Universal frontmatter
        frontmatter = {
            "ads_version": EDS_VERSION,
            "type": artifact_type,
            "experiment_id": experiment_id,
            "status": kwargs.get("status", "active"),
            "created": now,
            "updated": now,
            "tags": kwargs.get("tags", []),
        }

        # Type-specific fields
        if artifact_type == "assessment":
            frontmatter["phase"] = kwargs.get("phase", "analysis")
            frontmatter["priority"] = kwargs.get("priority", "medium")
            frontmatter["evidence_count"] = kwargs.get("evidence_count", 0)

        elif artifact_type == "report":
            frontmatter["metrics"] = kwargs.get("metrics", [])
            frontmatter["baseline"] = kwargs.get("baseline", "")
            frontmatter["comparison"] = kwargs.get("comparison", "baseline")

        elif artifact_type == "guide":
            frontmatter["commands"] = kwargs.get("commands", [])
            frontmatter["prerequisites"] = kwargs.get("prerequisites", [])

        elif artifact_type == "script":
            frontmatter["dependencies"] = kwargs.get("dependencies", [])

        # Generate YAML frontmatter
        yaml_lines = ["---"]
        for key, value in frontmatter.items():
            if isinstance(value, str):
                yaml_lines.append(f'{key}: "{value}"')
            elif isinstance(value, list):
                yaml_lines.append(f"{key}: {json.dumps(value)}")
            elif isinstance(value, (int, float)):
                yaml_lines.append(f"{key}: {value}")
        yaml_lines.append("---")

        # Generate content template
        content_sections = self._get_content_template(artifact_type)

        return "\n".join(yaml_lines) + "\n\n" + f"# {title}\n\n" + content_sections

    def _get_content_template(self, artifact_type: str) -> str:
        """Get content template for artifact type."""
        templates = {
            "assessment": """## Summary

Brief summary of assessment findings.

## Key Findings

1. Finding 1
2. Finding 2
3. Finding 3

## Evidence

### Evidence 1

Description and details.

### Evidence 2

Description and details.

## Recommendations

1. Recommendation 1
2. Recommendation 2

## Conclusion

Conclusion and next steps.
""",
            "report": """## Overview

Report overview and objectives.

## Metrics

| Metric | Value | Baseline | Change |
|--------|-------|----------|--------|
| Metric1 | 0.00 | 0.00 | +0.0% |
| Metric2 | 0.00 | 0.00 | +0.0% |

## Analysis

Detailed analysis of results.

## Comparison

Comparison with baseline/previous runs.

## Insights

Key insights and observations.

## Recommendations

1. Recommendation 1
2. Recommendation 2
""",
            "guide": """## Overview

Guide overview and purpose.

## Prerequisites

- Prerequisite 1
- Prerequisite 2

## Steps

### Step 1: Title

Instructions for step 1.

```bash
# Commands here
```

### Step 2: Title

Instructions for step 2.

```bash
# Commands here
```

## Validation

How to validate successful completion.

## Troubleshooting

Common issues and solutions.
""",
            "script": """## Overview

Script overview and purpose.

## Dependencies

- Dependency 1
- Dependency 2

## Usage

```bash
# Usage example
python script.py --arg value
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| --param1 | str | "" | Description |
| --param2 | int | 0 | Description |

## Output

Description of expected output.

## Example

```bash
# Example usage
python script.py --example
```

## Notes

Additional notes and considerations.
""",
        }

        return templates.get(artifact_type, "## Content\n\nAdd content here.\n")

    # ==================== STATE MANAGEMENT METHODS (NEW) ====================

    def _init_experiment_in_db(self, experiment_id: str, name: str, description: str, tags: list[str]):
        """Initialize experiment_state entry in database."""
        if not self.db_path.exists():
            return  # Database not available

        conn = sqlite3.connect(self.db_path)
        try:
            now = datetime.now(UTC).isoformat()

            # Insert into experiment_state table
            conn.execute(
                """
                INSERT INTO experiment_state (
                    experiment_id, current_task_id, current_phase, status,
                    created_at, updated_at
                ) VALUES (?, NULL, 'planning', 'active', ?, ?)
            """,
                (experiment_id, now, now),
            )

            conn.commit()
        except sqlite3.IntegrityError:
            # Experiment state already exists
            pass
        finally:
            conn.close()

    def get_current_state_from_db(self, experiment_id: str) -> dict | None:
        """Query current state from database (O(1) indexed query)."""
        if not self.db_path.exists():
            return None

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        try:
            cursor = conn.execute(
                """
                SELECT
                    s.current_task_id,
                    s.current_phase,
                    s.status,
                    t.title as current_task_title,
                    t.priority as current_task_priority
                FROM experiment_state s
                LEFT JOIN experiment_tasks t ON s.current_task_id = t.task_id
                WHERE s.experiment_id = ?
            """,
                (experiment_id,),
            )

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    def create_task(self, experiment_id: str, title: str, description: str = "", priority: str = "medium") -> str:
        """Create new task in database."""
        import uuid

        task_id = f"{title.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        now = datetime.now(UTC).isoformat()

        if not self.db_path.exists():
            raise RuntimeError("Database not available")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO experiment_tasks (
                    task_id, experiment_id, title, description,
                    status, priority, created_at
                ) VALUES (?, ?, ?, ?, 'backlog', ?, ?)
            """,
                (task_id, experiment_id, title, description, priority, now),
            )

            conn.commit()
            return task_id
        finally:
            conn.close()

    def set_current_task(self, experiment_id: str, task_id: str):
        """Atomically update current task with state transition logging."""
        if not self.db_path.exists():
            raise RuntimeError("Database not available")

        conn = sqlite3.connect(self.db_path)
        try:
            now = datetime.now(UTC).isoformat()

            # Get previous task
            cursor = conn.execute("SELECT current_task_id FROM experiment_state WHERE experiment_id = ?", (experiment_id,))
            row = cursor.fetchone()
            prev_task_id = row[0] if row else None

            # Update state
            conn.execute(
                """
                UPDATE experiment_state
                SET current_task_id = ?, updated_at = ?
                WHERE experiment_id = ?
            """,
                (task_id, now, experiment_id),
            )

            # Mark old task completed, new task in-progress
            if prev_task_id:
                conn.execute(
                    """
                    UPDATE experiment_tasks
                    SET status = 'completed', completed_at = ?
                    WHERE task_id = ?
                """,
                    (now, prev_task_id),
                )

            conn.execute(
                """
                UPDATE experiment_tasks
                SET status = 'in_progress', started_at = ?
                WHERE task_id = ?
            """,
                (now, task_id),
            )

            # Log transition
            conn.execute(
                """
                INSERT INTO state_transitions (
                    experiment_id, from_state, to_state,
                    transition_type, triggered_by, timestamp
                ) VALUES (?, ?, ?, 'task', 'etk_cli', ?)
            """,
                (experiment_id, prev_task_id or "none", task_id, now),
            )

            conn.commit()

            # Also update .state file
            exp_path = self.experiments_dir / experiment_id
            if exp_path.exists():
                update_state(exp_path, current_task=task_id)

        finally:
            conn.close()

    def record_decision(self, experiment_id: str, decision: str, rationale: str, impact: str = None) -> str:
        """Log decision to database."""
        import uuid

        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        if not self.db_path.exists():
            raise RuntimeError("Database not available")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO experiment_decisions (
                    decision_id, experiment_id, date, decision, rationale, impact, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (decision_id, experiment_id, now.date().isoformat(), decision, rationale, impact, now.isoformat()),
            )

            conn.commit()
            return decision_id
        finally:
            conn.close()

    def record_insight(self, experiment_id: str, insight: str, impact: str, category: str = "observation") -> str:
        """Log insight to database."""
        import uuid

        insight_id = f"ins_{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)

        if not self.db_path.exists():
            raise RuntimeError("Database not available")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO experiment_insights (
                    insight_id, experiment_id, date, insight, impact, category, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (insight_id, experiment_id, now.date().isoformat(), insight, impact, category, now.isoformat()),
            )

            conn.commit()
            return insight_id
        finally:
            conn.close()


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
                if args.phase:
                    kwargs["phase"] = args.phase
                if args.priority:
                    kwargs["priority"] = args.priority

            elif args.type == "report":
                if args.metrics:
                    kwargs["metrics"] = [m.strip() for m in args.metrics.split(",")]
                if args.baseline:
                    kwargs["baseline"] = args.baseline

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
                if line.strip():
                    print(line)

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
                print(
                    f"   Artifacts: {total} (A:{exp['artifacts']['assessments']} R:{exp['artifacts']['reports']} G:{exp['artifacts']['guides']} S:{exp['artifacts']['scripts']})"
                )
                print()

        elif args.command == "sync":
            print("🔄 Syncing artifacts to database...")
            stats = tracker.sync_to_database(args.experiment_id, args.all)

            print("\n✅ Sync complete:")
            print(f"   ✓ Synced: {stats['synced']}")
            if stats["skipped"] > 0:
                print(f"   ⊘ Skipped: {stats['skipped']}")
            if stats["failed"] > 0:
                print(f"   ✗ Failed: {stats['failed']}")

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

            # Experiments section
            exp = analytics["experiments"]
            print("🧪 Experiments")
            print(f"   Total: {exp['total']}")
            print(f"   Active: {exp['active']}")
            print(f"   Complete: {exp['complete']}")
            print(f"   Deprecated: {exp['deprecated']}")

            # Artifacts section
            print(f"\n📋 Artifacts: {analytics['total_artifacts']} total")
            for artifact_type, count in analytics["artifacts_by_type"].items():
                print(f"   {artifact_type}: {count}")

            # Artifacts per experiment
            print("\n📦 Artifacts per Experiment")
            for exp_data in analytics["artifacts_per_experiment"][:5]:
                print(f"   {exp_data['name']}: {exp_data['artifact_count']}")

            # Popular tags
            if analytics["popular_tags"]:
                print("\n🏷️  Popular Tags")
                for tag_data in analytics["popular_tags"][:5]:
                    print(f"   {tag_data['tag_name']}: {tag_data['artifact_count']} artifacts, {tag_data['experiment_count']} experiments")

            # Recent activity
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
