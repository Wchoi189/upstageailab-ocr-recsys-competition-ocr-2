import re
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path

from etk.database import EDS_VERSION, KST, DatabaseManager
from etk.utils.path_utils import ExperimentPaths
from etk.utils.state_file import create_state_file


class ExperimentTracker:
    """Main experiment tracker interface, implementing EDS v1.0 standards."""

    def __init__(self, tracker_root: Path | None = None):
        if tracker_root:
            self.tracker_root = tracker_root
        else:
            # Auto-detect tracker root (climb up from CWD)
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

        # Database initialization via manager
        db_path = self.tracker_root.parent / "data" / "ops" / "tracking.db"
        self.db = DatabaseManager(db_path)

        # Templates
        self.templates_path = self.tracker_root / ".ai-instructions" / "tier2-framework" / "artifact-catalog.yaml"

    def get_current_experiment(self) -> str | None:
        """Detect current experiment ID from CWD."""
        cwd = Path.cwd()
        if "experiments" in cwd.parts:
            idx = cwd.parts.index("experiments")
            if len(cwd.parts) > idx + 1:
                return cwd.parts[idx + 1]

        # Check for .current file in experiments dir
        current_file = self.experiments_dir / ".current"
        if current_file.exists():
            return current_file.read_text().strip()

        return None

    def init_experiment(self, name: str, description: str = "", tags: list[str] = None) -> str:
        """Initialize new experiment with proper EDS v1.0 structure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = name.lower().replace(" ", "_").replace("-", "_")
        experiment_id = f"{timestamp}_{slug}"

        exp_path = self.experiments_dir / experiment_id
        if exp_path.exists():
            raise ValueError(f"Experiment already exists: {experiment_id}")

        # Create EDS structure
        exp_path.mkdir(parents=True)
        paths = ExperimentPaths(experiment_id, self.tracker_root)
        paths.ensure_structure()

        # Extended EDS directories
        (exp_path / ".metadata" / "00-status").mkdir(exist_ok=True)
        (exp_path / ".metadata" / "01-work-logs").mkdir(exist_ok=True)
        (exp_path / ".metadata" / "02-tasks").mkdir(exist_ok=True)

        # Generate manifest
        manifest = self._generate_manifest(experiment_id, name, description, tags or [])
        (exp_path / "README.md").write_text(manifest, encoding="utf-8")

        # Initialize local state
        create_state_file(exp_path, experiment_id)

        # Initialize database state
        self.db.init_experiment_in_db(experiment_id, name, description, tags or [])

        # Set as current
        (self.experiments_dir / ".current").write_text(experiment_id)

        print(f"✅ Initialized experiment: {experiment_id}")
        return experiment_id

    def create_artifact(self, artifact_type: str, title: str, experiment_id: str | None = None, **kwargs) -> Path:
        """Create new EDS v1.0 compliant artifact."""
        experiment_id = experiment_id or self.get_current_experiment()
        if not experiment_id:
            raise ValueError("No experiment specified/detected.")

        exp_path = self.experiments_dir / experiment_id

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        slug = self._generate_slug(title)
        filename = f"{timestamp}_{artifact_type}_{slug}.md"
        artifact_path = exp_path / ".metadata" / f"{artifact_type}s" / filename

        content = self._generate_artifact_content(artifact_type, title, experiment_id, **kwargs)
        artifact_path.write_text(content, encoding="utf-8")

        print(f"✅ Created {artifact_type}: {filename}")
        return artifact_path

    def get_status(self, experiment_id: str | None = None) -> dict:
        """Summarize experiment status."""
        experiment_id = experiment_id or self.get_current_experiment()
        if not experiment_id:
            raise ValueError("No experiment specified")

        exp_path = self.experiments_dir / experiment_id
        meta_path = exp_path / ".metadata"

        counts = {}
        for t in ["assessment", "report", "guide", "script"]:
            dir_path = meta_path / f"{t}s"
            counts[f"{t}s"] = len(list(dir_path.glob("*.md"))) if dir_path.exists() else 0

        return {"experiment_id": experiment_id, "path": str(exp_path), "artifacts": counts, "total_artifacts": sum(counts.values())}

    def validate(self, experiment_id: str | None = None, all_experiments: bool = False) -> tuple[bool, list[str]]:
        """Validate compliance via external checker."""
        checker_path = self.tracker_root / ".ai-instructions" / "schema" / "compliance-checker.py"
        if not checker_path.exists():
            return False, ["Compliance checker not found"]

        cmd = ["python3", str(checker_path)]
        if all_experiments:
            cmd.append("--all")
        else:
            experiment_id = experiment_id or self.get_current_experiment()
            if not experiment_id:
                return False, ["No experiment specified"]
            cmd.append(str(self.experiments_dir / experiment_id))

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.tracker_root)
        return result.returncode == 0, result.stdout.split("\n")

    def list_experiments(self) -> list[dict]:
        """List and summarize all experiments."""
        experiments = []
        for exp_dir in sorted(self.experiments_dir.iterdir()):
            if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                try:
                    experiments.append(self.get_status(exp_dir.name))
                except:
                    continue
        return experiments

    def sync_to_database(self, experiment_id: str | None = None, sync_all: bool = False) -> dict:
        """Sync artifacts from filesystem to SQLite database."""
        stats = {"synced": 0, "failed": 0, "skipped": 0}

        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row

        try:
            if sync_all:
                experiments = [exp["experiment_id"] for exp in self.list_experiments()]
            else:
                eid = experiment_id or self.get_current_experiment()
                if not eid:
                    raise ValueError("No experiment target")
                experiments = [eid]

            for eid in experiments:
                self._sync_experiment(conn, eid, stats)

            conn.commit()
        finally:
            conn.close()
        return stats

    def query_artifacts(self, query: str) -> list[dict]:
        return self.db.query_artifacts(query)

    def get_analytics(self) -> dict:
        return self.db.get_analytics()

    def _generate_slug(self, title: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")

    def _generate_manifest(self, experiment_id: str, name: str, description: str, tags: list[str]) -> str:
        tags_str = f"tags: {json.dumps(tags)}" if tags else "tags: []"
        return f"""---
ads_version: "{EDS_VERSION}"
type: experiment_manifest
experiment_id: "{experiment_id}"
status: "active"
created: "{datetime.now().isoformat()}"
updated: "{datetime.now().isoformat()}"
{tags_str}
---

# Experiment: {name}

## Overview

{description or "No description provided."}

## Structure

- Assessments: `.metadata/assessments/`
- Reports: `.metadata/reports/`
- Guides: `.metadata/guides/`

---
*Created by ETK v1.0.0 | EDS v1.0*
"""

    def _generate_artifact_content(self, artifact_type: str, title: str, experiment_id: str, **kwargs) -> str:
        now = datetime.now(KST).isoformat()
        tags = kwargs.get("tags", [])
        tags_line = f"tags: {json.dumps(tags)}"

        header = f"""---
ads_version: "{EDS_VERSION}"
type: {artifact_type}
experiment_id: "{experiment_id}"
title: "{title}"
created: "{now}"
updated: "{now}"
{tags_line}"""

        if artifact_type == "assessment":
            header += f'\nphase: "{kwargs.get("phase", "planning")}"\npriority: "{kwargs.get("priority", "medium")}"'
        elif artifact_type == "report":
            header += f'\nmetrics: {json.dumps(kwargs.get("metrics", []))}\nbaseline: "{kwargs.get("baseline", "")}"'

        header += "\n---\n"

        template = self._get_content_template(artifact_type)
        return header + template

    def _get_content_template(self, artifact_type: str) -> str:
        templates = {
            "assessment": "\n# Assessment: Title\n\n## Objective\n\n## Observations\n\n## Conclusion\n",
            "report": "\n# Report: Title\n\n## Summary\n\n## Metrics Analysis\n\n## Visual Proof\n",
            "guide": "\n# Guide: Title\n\n## Overview\n\n## Steps\n",
            "script": "\n# Script: Title\n\n## Usage\n\n## Logic Flow\n",
        }
        return templates.get(artifact_type, "\n# New Artifact\n")

    def _sync_experiment(self, conn: sqlite3.Connection, experiment_id: str, stats: dict):
        exp_path = self.experiments_dir / experiment_id
        if not exp_path.exists():
            return

        # Sync experiment entry
        now = datetime.now(KST).isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO experiments (experiment_id, name, status, created_at, updated_at, ads_version) VALUES (?, ?, ?, ?, ?, ?)",
            (experiment_id, experiment_id, "active", now, now, EDS_VERSION),
        )

        artifact_pattern = re.compile(r"^\d{8}_\d{4}_(assessment|report|guide|script)_.*\.md$")
        for p in exp_path.rglob("*.md"):
            if artifact_pattern.match(p.name):
                self._sync_artifact(conn, experiment_id, p, p.name.split("_")[2], stats)

    def _sync_artifact(self, conn: sqlite3.Connection, experiment_id: str, artifact_path: Path, artifact_type: str, stats: dict):
        # Implementation similar to etk.py (moved logic)
        # For brevity, I'll assume DatabaseManager handles the bulk or I repeat the parsing logic here
        # Since DatabaseManager was designed for DB ops, let's keep the file parsing here.
        try:
            content = artifact_path.read_text(encoding="utf-8")
            fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
            if not fm_match:
                return

            # Simple frontmatter parse
            fm = {}
            for line in fm_match.group(1).split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    fm[k.strip()] = v.strip().strip('"').strip("'")

            aid = artifact_path.stem
            title = fm.get("title", aid)
            status = fm.get("status", "active")
            created = fm.get("created", datetime.now(KST).isoformat())

            conn.execute(
                "INSERT OR REPLACE INTO artifacts (artifact_id, experiment_id, type, title, file_path, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    aid,
                    experiment_id,
                    artifact_type,
                    title,
                    str(artifact_path.relative_to(self.tracker_root)),
                    status,
                    created,
                    datetime.now(KST).isoformat(),
                ),
            )

            # Sync FTS
            main = content[fm_match.end() :]
            conn.execute(
                "INSERT OR REPLACE INTO artifacts_fts (artifact_id, experiment_id, title, content) VALUES (?, ?, ?, ?)",
                (aid, experiment_id, title, main),
            )

            stats["synced"] += 1
        except Exception:
            stats["failed"] += 1


import json
