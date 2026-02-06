import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

# Define KST timezone (UTC+9)
KST = timezone(timedelta(hours=9))
EDS_VERSION = "1.0"


class DatabaseManager:
    """Handles SQLite database operations for experiment tracking."""

    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _get_connection(self):
        """Get a database connection."""
        if not self.db_path.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_experiment_in_db(self, experiment_id: str, name: str, description: str, tags: list[str]):
        """Initialize experiment_state entry in database."""
        conn = self._get_connection()
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
            pass
        finally:
            conn.close()

    def get_current_state(self, experiment_id: str) -> dict | None:
        """Query current state from database."""
        conn = self._get_connection()
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
            return dict(row) if row else None
        finally:
            conn.close()

    def create_task(self, experiment_id: str, title: str, description: str = "", priority: str = "medium") -> str:
        """Create new task in database."""
        import uuid

        task_id = f"{title.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
        now = datetime.now(UTC).isoformat()
        conn = self._get_connection()
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

    def update_current_task(self, experiment_id: str, task_id: str, trigger: str = "etk_cli"):
        """Atomically update current task and log transition."""
        conn = self._get_connection()
        try:
            now = datetime.now(UTC).isoformat()
            cursor = conn.execute("SELECT current_task_id FROM experiment_state WHERE experiment_id = ?", (experiment_id,))
            row = cursor.fetchone()
            prev_task_id = row[0] if row else None

            conn.execute(
                "UPDATE experiment_state SET current_task_id = ?, updated_at = ? WHERE experiment_id = ?",
                (task_id, now, experiment_id),
            )

            if prev_task_id:
                conn.execute(
                    "UPDATE experiment_tasks SET status = 'completed', completed_at = ? WHERE task_id = ?",
                    (now, prev_task_id),
                )

            conn.execute(
                "UPDATE experiment_tasks SET status = 'in_progress', started_at = ? WHERE task_id = ?",
                (now, task_id),
            )

            conn.execute(
                """
                INSERT INTO state_transitions (
                    experiment_id, from_state, to_state,
                    transition_type, triggered_by, timestamp
                ) VALUES (?, ?, ?, 'task', ?, ?)
            """,
                (experiment_id, prev_task_id or "none", task_id, trigger, now),
            )
            conn.commit()
        finally:
            conn.close()

    def record_decision(self, experiment_id: str, decision: str, rationale: str, impact: str | None = None) -> str:
        """Log decision to database."""
        import uuid

        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)
        conn = self._get_connection()
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
        conn = self._get_connection()
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

    def query_artifacts(self, query: str, limit: int = 20) -> list[dict]:
        """Search artifacts using FTS5."""
        conn = self._get_connection()
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
                LIMIT ?
            """,
                (query, limit),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_analytics(self) -> dict:
        """Generate analytics dashboard data."""
        from typing import Any

        conn = self._get_connection()
        try:
            analytics: dict[str, Any] = {}
            # Experiment counts
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) as complete,
                    SUM(CASE WHEN status = 'deprecated' THEN 1 ELSE 0 END) as deprecated
                FROM experiments
            """).fetchone()
            analytics["experiments"] = dict(row)

            # Artifact counts by type
            cursor = conn.execute("""
                SELECT type, COUNT(*) as count
                FROM artifacts
                GROUP BY type
                ORDER BY count DESC
            """)
            analytics["artifacts_by_type"] = {row["type"]: row["count"] for row in cursor.fetchall()}

            # Total artifacts
            analytics["total_artifacts"] = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]

            # Recent activity
            cursor = conn.execute("""
                SELECT a.artifact_id, a.title, a.type, a.updated_at, e.name as experiment_name
                FROM artifacts a
                JOIN experiments e ON a.experiment_id = e.experiment_id
                ORDER BY a.updated_at DESC LIMIT 10
            """)
            analytics["recent_activity"] = [dict(row) for row in cursor.fetchall()]

            # Popular tags
            cursor = conn.execute("""
                SELECT t.tag_name, COUNT(DISTINCT at.artifact_id) as artifact_count, COUNT(DISTINCT t.experiment_id) as experiment_count
                FROM tags t
                LEFT JOIN artifact_tags at ON t.tag_name = at.tag_name
                GROUP BY t.tag_name ORDER BY artifact_count DESC LIMIT 10
            """)
            analytics["popular_tags"] = [dict(row) for row in cursor.fetchall()]

            return analytics
        finally:
            conn.close()
