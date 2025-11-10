#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def _load_bootstrap():
    if "scripts._bootstrap" in sys.modules:
        return sys.modules["scripts._bootstrap"]

    current_dir = Path(__file__).resolve().parent
    for directory in (current_dir, *tuple(current_dir.parents)):
        candidate = directory / "_bootstrap.py"
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(
                "scripts._bootstrap", candidate
            )
            if spec is None or spec.loader is None:  # pragma: no cover - defensive
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    raise RuntimeError("Could not locate scripts/_bootstrap.py")


_BOOTSTRAP = _load_bootstrap()
setup_project_paths = _BOOTSTRAP.setup_project_paths
get_path_resolver = _BOOTSTRAP.get_path_resolver

setup_project_paths()
REPO_ROOT = get_path_resolver().project_root
VERSION_FILE = REPO_ROOT / "project_version.yaml"
CHANGELOG_DRAFT = REPO_ROOT / "CHANGELOG.draft.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"


def get_version_info() -> dict:
    if not VERSION_FILE.exists():
        return {"version": "0.0.0", "release_date": None}
    return yaml.safe_load(VERSION_FILE.read_text(encoding="utf-8")) or {}


def get_last_version() -> str | None:
    """Extract last version from CHANGELOG.md."""
    if not CHANGELOG.exists():
        return None
    content = CHANGELOG.read_text(encoding="utf-8")
    # Look for first version entry: ## [YYYY-MM-DD] - Title
    lines = content.split("\n")
    for line in lines:
        if line.startswith("## ["):
            # Extract date from [YYYY-MM-DD]
            date_part = line.split("[")[1].split("]")[0] if "[" in line else None
            return date_part
    return None


def get_completed_plans(since_date: str | None = None) -> list[dict]:
    """Query tracking DB for completed plans since date."""
    try:
        from scripts.agent_tools.utilities.tracking.db import (
            DB_PATH,
            get_connection,
            init_db,
        )

        # Auto-initialize DB if it doesn't exist or is empty
        if not DB_PATH.exists() or (DB_PATH.exists() and DB_PATH.stat().st_size == 0):
            init_db()
        conn = get_connection()
        # Try query - if tables don't exist, initialize and retry
        try:
            query = """
                SELECT key, title, status, updated_at, owner
                FROM feature_plans
                WHERE status = 'completed'
            """
            params = []
            if since_date:
                query += " AND updated_at >= ?"
                params.append(since_date)
            query += " ORDER BY updated_at DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            # Tables don't exist - initialize and return empty
            init_db()
            return []
    except Exception as e:
        print(f"Warning: Could not query tracking DB: {e}")
        return []


def get_completed_experiments(since_date: str | None = None) -> list[dict]:
    """Query tracking DB for completed experiments with summaries."""
    try:
        from scripts.agent_tools.utilities.tracking.db import (
            DB_PATH,
            get_connection,
            init_db,
        )

        # Auto-initialize DB if it doesn't exist or is empty
        if not DB_PATH.exists() or (DB_PATH.exists() and DB_PATH.stat().st_size == 0):
            init_db()
        conn = get_connection()
        # Try query - if tables don't exist, initialize and retry
        try:
            query = """
                SELECT e.key, e.title, e.status, e.updated_at, e.objective,
                       COUNT(r.id) as run_count
                FROM experiments e
                LEFT JOIN experiment_runs r ON r.experiment_id = e.id
                WHERE e.status = 'completed'
            """
            params = []
            if since_date:
                query += " AND e.updated_at >= ?"
                params.append(since_date)
            query += " GROUP BY e.id ORDER BY e.updated_at DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            # Tables don't exist - initialize and return empty
            init_db()
            return []
    except Exception as e:
        print(f"Warning: Could not query tracking DB: {e}")
        return []


def get_git_commits(since_date: str | None = None) -> list[dict]:
    """Parse git log for conventional commits."""
    try:
        cmd = ["git", "log", "--pretty=format:%H|%ad|%s", "--date=iso"]
        if since_date:
            cmd.append(f"--since={since_date}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
        if result.returncode != 0:
            return []
        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            hash_val, date, message = parts
            # Parse conventional commit format: type(scope): description
            commit_type = None
            scope = None
            description = message
            if ":" in message:
                prefix, description = message.split(":", 1)
                description = description.strip()
                if "(" in prefix and ")" in prefix:
                    commit_type, scope = prefix.split("(")
                    scope = scope.rstrip(")")
                else:
                    commit_type = prefix
            commits.append(
                {
                    "hash": hash_val,
                    "date": date,
                    "message": message,
                    "type": commit_type,
                    "scope": scope,
                    "description": description,
                }
            )
        return commits
    except Exception as e:
        print(f"Warning: Could not parse git log: {e}")
        return []


def get_deprecated_docs() -> list[dict]:
    """Get recently deprecated docs from version bump."""
    try:
        from scripts.agent_tools.documentation.deprecate_docs import get_docs_health

        version_info = get_version_info()
        current_version = str(version_info.get("version", "0.0.0"))
        get_docs_health(current_version)
        # Return docs that were just deprecated (in _archived)
        archived_dir = REPO_ROOT / "docs" / "artifacts" / "_archived"
        if not archived_dir.exists():
            return []
        deprecated = []
        for md_path in archived_dir.rglob("*.md"):
            # Check modification time (recently moved)
            mtime = datetime.fromtimestamp(md_path.stat().st_mtime)
            if (datetime.now() - mtime).days < 7:  # Last 7 days
                deprecated.append(
                    {
                        "path": str(md_path.relative_to(REPO_ROOT)),
                        "title": md_path.stem,
                    }
                )
        return deprecated
    except Exception as e:
        print(f"Warning: Could not get deprecated docs: {e}")
        return []


def format_date(date_str: str | None) -> str:
    """Format date for changelog entry."""
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        # Parse ISO date and format
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def generate_draft() -> str:
    """Generate changelog draft from all sources."""
    version_info = get_version_info()
    current_version = str(version_info.get("version", "0.0.0"))
    release_date = version_info.get("release_date")
    notes = version_info.get("notes", "")

    date_str = format_date(release_date)
    last_date = get_last_version()

    lines = []
    lines.append(f"## [{date_str}] - Version {current_version}")
    if notes:
        lines.append(f"\n{notes}\n")

    # Collect data
    plans = get_completed_plans(last_date)
    experiments = get_completed_experiments(last_date)
    commits = get_git_commits(last_date)
    deprecated = get_deprecated_docs()

    # Group commits by type
    feat_commits = [c for c in commits if c.get("type") == "feat"]
    fix_commits = [c for c in commits if c.get("type") == "fix"]
    refactor_commits = [c for c in commits if c.get("type") == "refactor"]
    other_commits = [
        c for c in commits if c.get("type") not in ("feat", "fix", "refactor")
    ]

    # Features (from plans and feat commits)
    if plans or feat_commits:
        lines.append("\n### ✅ Features")
        for plan in plans:
            lines.append(f"- **{plan['title']}** ({plan['key']})")
            if plan.get("owner"):
                lines.append(f"  - Owner: {plan['owner']}")
        for commit in feat_commits:
            lines.append(f"- {commit['description']}")
            if commit.get("scope"):
                lines.append(f"  - Scope: {commit['scope']}")

    # Experiments
    if experiments:
        lines.append("\n### 🧪 Experiments")
        for exp in experiments:
            lines.append(f"- **{exp['title']}** ({exp['key']})")
            if exp.get("objective"):
                lines.append(f"  - Objective: {exp['objective']}")
            if exp.get("run_count"):
                lines.append(f"  - Runs: {exp['run_count']}")

    # Fixes
    if fix_commits:
        lines.append("\n### 🐛 Fixes")
        for commit in fix_commits:
            lines.append(f"- {commit['description']}")
            if commit.get("scope"):
                lines.append(f"  - Scope: {commit['scope']}")

    # Refactoring
    if refactor_commits:
        lines.append("\n### 🔧 Refactoring")
        for commit in refactor_commits:
            lines.append(f"- {commit['description']}")
            if commit.get("scope"):
                lines.append(f"  - Scope: {commit['scope']}")

    # Documentation
    if deprecated:
        lines.append("\n### 📚 Documentation")
        lines.append("- Deprecated outdated documentation:")
        for doc in deprecated:
            lines.append(f"  - {doc['title']} ({doc['path']})")

    # Other changes
    if other_commits:
        lines.append("\n### 📋 Other Changes")
        for commit in other_commits:
            commit_type = commit.get("type", "change")
            lines.append(f"- [{commit_type}] {commit['description']}")

    # Summary
    total_items = len(plans) + len(experiments) + len(commits) + len(deprecated)
    if total_items > 0:
        lines.append("\n### 📊 Summary")
        lines.append(f"- Completed Plans: {len(plans)}")
        lines.append(f"- Experiments: {len(experiments)}")
        lines.append(f"- Git Commits: {len(commits)}")
        lines.append(f"- Deprecated Docs: {len(deprecated)}")
        lines.append(f"\n**Total**: {total_items} items")

    lines.append("\n---")
    lines.append(
        "\n*This draft was automatically generated. Review and edit before merging into CHANGELOG.md*"
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate changelog draft from tracking DB and git log"
    )
    parser.add_argument(
        "--output", default=str(CHANGELOG_DRAFT), help="Output file path"
    )
    parser.add_argument(
        "--preview", action="store_true", help="Print to stdout instead of file"
    )
    args = parser.parse_args()

    try:
        draft = generate_draft()
        if args.preview:
            print(draft)
        else:
            output_path = Path(args.output)
            output_path.write_text(draft, encoding="utf-8")
            print(f"✅ Generated changelog draft: {output_path}")
            print("📝 Review and edit before merging into CHANGELOG.md")
        return 0
    except Exception as e:
        print(f"❌ Error generating draft: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
