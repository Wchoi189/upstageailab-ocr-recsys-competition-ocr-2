from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from AgentQMS.agent_tools.utils.paths import get_project_root
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

DB_PATH = get_project_root() / "data/ops/tracking.db"


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def get_connection(readonly: bool = False) -> sqlite3.Connection:
    dsn = DB_PATH
    if readonly and not dsn.exists():
        raise FileNotFoundError(f"Tracking DB not found: {dsn}")
    _ensure_parent_dir(dsn)
    conn = sqlite3.connect(str(dsn))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    with conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        status_check = "CHECK(status IN ('pending','in_progress','paused','completed','cancelled'))"

        conn.execute(
            f"""
			CREATE TABLE IF NOT EXISTS feature_plans (
				id INTEGER PRIMARY KEY,
				key TEXT UNIQUE NOT NULL,
				title TEXT NOT NULL,
				status TEXT NOT NULL DEFAULT 'pending' {status_check},
				owner TEXT,
				started_at TEXT,
				updated_at TEXT
			);
			"""
        )

        conn.execute(
            f"""
			CREATE TABLE IF NOT EXISTS plan_tasks (
				id INTEGER PRIMARY KEY,
				plan_id INTEGER NOT NULL,
				title TEXT NOT NULL,
				status TEXT NOT NULL DEFAULT 'pending' {status_check},
				notes TEXT,
				created_at TEXT NOT NULL,
				updated_at TEXT,
				FOREIGN KEY(plan_id) REFERENCES feature_plans(id) ON DELETE CASCADE
			);
			"""
        )

        conn.execute(
            f"""
			CREATE TABLE IF NOT EXISTS refactors (
				id INTEGER PRIMARY KEY,
				key TEXT UNIQUE NOT NULL,
				title TEXT NOT NULL,
				status TEXT NOT NULL DEFAULT 'pending' {status_check},
				notes TEXT,
				started_at TEXT,
				updated_at TEXT
			);
			"""
        )

        conn.execute(
            f"""
			CREATE TABLE IF NOT EXISTS debug_sessions (
				id INTEGER PRIMARY KEY,
				key TEXT UNIQUE NOT NULL,
				title TEXT NOT NULL,
				status TEXT NOT NULL DEFAULT 'in_progress' {status_check},
				hypothesis TEXT,
				scope TEXT,
				started_at TEXT NOT NULL,
				updated_at TEXT
			);
			"""
        )

        conn.execute(
            """
			CREATE TABLE IF NOT EXISTS debug_notes (
				id INTEGER PRIMARY KEY,
				session_id INTEGER NOT NULL,
				note TEXT NOT NULL,
				created_at TEXT NOT NULL,
				FOREIGN KEY(session_id) REFERENCES debug_sessions(id) ON DELETE CASCADE
			);
			"""
        )

        conn.execute(
            f"""
			CREATE TABLE IF NOT EXISTS experiments (
				id INTEGER PRIMARY KEY,
				key TEXT UNIQUE NOT NULL,
				title TEXT NOT NULL,
				objective TEXT,
				owner TEXT,
				status TEXT NOT NULL DEFAULT 'in_progress' {status_check},
				created_at TEXT NOT NULL,
				updated_at TEXT
			);
			"""
        )

        conn.execute(
            """
			CREATE TABLE IF NOT EXISTS experiment_runs (
				id INTEGER PRIMARY KEY,
				experiment_id INTEGER NOT NULL,
				run_no INTEGER NOT NULL,
				params_json TEXT NOT NULL,
				metrics_json TEXT,
				outcome TEXT NOT NULL CHECK(outcome IN ('pass','fail','inconclusive')),
				created_at TEXT NOT NULL,
				FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
				UNIQUE(experiment_id, run_no)
			);
			"""
        )

        conn.execute(
            """
			CREATE TABLE IF NOT EXISTS experiment_artifacts (
				id INTEGER PRIMARY KEY,
				experiment_id INTEGER NOT NULL,
				run_id INTEGER,
				type TEXT NOT NULL,
				path TEXT NOT NULL,
				created_at TEXT NOT NULL,
				FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
				FOREIGN KEY(run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
			);
			"""
        )

        conn.execute(
            """
			CREATE TABLE IF NOT EXISTS summaries (
				id INTEGER PRIMARY KEY,
				entity_type TEXT NOT NULL,
				entity_id INTEGER NOT NULL,
				style TEXT NOT NULL,
				text TEXT NOT NULL,
				created_at TEXT NOT NULL
			);
			"""
        )


# Feature plans
def upsert_feature_plan(key: str, title: str, owner: str | None = None) -> int:
    conn = get_connection()
    with conn:
        row = conn.execute(
            "SELECT id FROM feature_plans WHERE key=?", (key,)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE feature_plans SET title=?, owner=?, updated_at=? WHERE id=?",
                (title, owner, _utc_now_iso(), row[0]),
            )
            return int(row[0])
        conn.execute(
            "INSERT INTO feature_plans(key,title,status,owner,started_at,updated_at) VALUES (?,?,?,?,?,?)",
            (key, title, "pending", owner, None, _utc_now_iso()),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def set_plan_status(key: str, status: str) -> None:
    allowed = {"pending", "in_progress", "paused", "completed", "cancelled"}
    if status not in allowed:
        raise ValueError(f"Invalid status: {status}")
    conn = get_connection()
    with conn:
        plan = conn.execute(
            "SELECT id FROM feature_plans WHERE key=?", (key,)
        ).fetchone()
        if not plan:
            raise KeyError(f"Plan not found: {key}")
        fields = ["updated_at = ?", _utc_now_iso()]
        if status == "in_progress":
            fields[0] = "started_at = ?, updated_at = ?"
            params: tuple[Any, ...] = (_utc_now_iso(), _utc_now_iso(), key)
        else:
            params = (_utc_now_iso(), key)
        conn.execute(
            f"UPDATE feature_plans SET status='{status}', {fields[0]} WHERE key=?",
            params,
        )


def add_plan_task(plan_key: str, title: str, notes: str | None = None) -> int:
    conn = get_connection()
    with conn:
        plan = conn.execute(
            "SELECT id FROM feature_plans WHERE key=?", (plan_key,)
        ).fetchone()
        if not plan:
            raise KeyError(f"Plan not found: {plan_key}")
        conn.execute(
            "INSERT INTO plan_tasks(plan_id,title,status,notes,created_at,updated_at) VALUES (?,?,?,?,?,?)",
            (int(plan[0]), title, "pending", notes, _utc_now_iso(), None),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def set_task_done(task_id: int) -> None:
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE plan_tasks SET status='completed', updated_at=? WHERE id=?",
            (_utc_now_iso(), task_id),
        )


# Refactors
def upsert_refactor(key: str, title: str, notes: str | None = None) -> int:
    conn = get_connection()
    with conn:
        row = conn.execute("SELECT id FROM refactors WHERE key=?", (key,)).fetchone()
        if row:
            conn.execute(
                "UPDATE refactors SET title=?, notes=?, updated_at=? WHERE id=?",
                (title, notes, _utc_now_iso(), row[0]),
            )
            return int(row[0])
        conn.execute(
            "INSERT INTO refactors(key,title,status,notes,started_at,updated_at) VALUES (?,?,?,?,?,?)",
            (key, title, "pending", notes, None, _utc_now_iso()),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def set_refactor_status(key: str, status: str) -> None:
    allowed = {"pending", "in_progress", "paused", "completed", "cancelled"}
    if status not in allowed:
        raise ValueError(f"Invalid status: {status}")
    conn = get_connection()
    with conn:
        ref = conn.execute("SELECT id FROM refactors WHERE key=?", (key,)).fetchone()
        if not ref:
            raise KeyError(f"Refactor not found: {key}")
        fields = ["updated_at = ?", _utc_now_iso()]
        if status == "in_progress":
            fields[0] = "started_at = ?, updated_at = ?"
            params = (_utc_now_iso(), _utc_now_iso(), key)
        else:
            params = (_utc_now_iso(), key)
        conn.execute(
            f"UPDATE refactors SET status='{status}', {fields[0]} WHERE key=?", params
        )


# Debugging
def create_debug_session(
    key: str, title: str, hypothesis: str = "", scope: str = ""
) -> int:
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO debug_sessions(key,title,status,hypothesis,scope,started_at,updated_at) VALUES (?,?,?,?,?,?,?)",
            (key, title, "in_progress", hypothesis, scope, _utc_now_iso(), None),
        )
        row = conn.execute(
            "SELECT id FROM debug_sessions WHERE key=?", (key,)
        ).fetchone()
        return int(row[0])


def add_debug_note(session_key: str, note: str) -> int:
    conn = get_connection()
    with conn:
        session = conn.execute(
            "SELECT id FROM debug_sessions WHERE key=?", (session_key,)
        ).fetchone()
        if not session:
            raise KeyError(f"Debug session not found: {session_key}")
        conn.execute(
            "INSERT INTO debug_notes(session_id,note,created_at) VALUES (?,?,?)",
            (int(session[0]), note, _utc_now_iso()),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def set_debug_status(key: str, status: str) -> None:
    allowed = {"pending", "in_progress", "paused", "completed", "cancelled"}
    if status not in allowed:
        raise ValueError(f"Invalid status: {status}")
    conn = get_connection()
    with conn:
        conn.execute(
            "UPDATE debug_sessions SET status=?, updated_at=? WHERE key=?",
            (status, _utc_now_iso(), key),
        )


# Experiments
def upsert_experiment(
    key: str, title: str, objective: str = "", owner: str | None = None
) -> int:
    conn = get_connection()
    with conn:
        row = conn.execute("SELECT id FROM experiments WHERE key=?", (key,)).fetchone()
        if row:
            conn.execute(
                "UPDATE experiments SET title=?, objective=?, owner=?, updated_at=? WHERE id=?",
                (title, objective, owner, _utc_now_iso(), row[0]),
            )
            return int(row[0])
        conn.execute(
            "INSERT INTO experiments(key,title,objective,owner,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
            (key, title, objective, owner, "in_progress", _utc_now_iso(), None),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def add_experiment_run(
    experiment_key: str,
    run_no: int,
    params: dict[str, Any],
    metrics: dict[str, Any] | None,
    outcome: str,
) -> int:
    if outcome not in {"pass", "fail", "inconclusive"}:
        raise ValueError("Invalid outcome")
    conn = get_connection()
    with conn:
        exp = conn.execute(
            "SELECT id FROM experiments WHERE key=?", (experiment_key,)
        ).fetchone()
        if not exp:
            raise KeyError(f"Experiment not found: {experiment_key}")
        conn.execute(
            "INSERT OR REPLACE INTO experiment_runs(experiment_id,run_no,params_json,metrics_json,outcome,created_at) VALUES (?,?,?,?,?,?)",
            (
                int(exp[0]),
                int(run_no),
                json.dumps(params, ensure_ascii=False),
                json.dumps(metrics or {}, ensure_ascii=False),
                outcome,
                _utc_now_iso(),
            ),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def link_experiment_artifact(
    experiment_key: str, type_: str, path: str, run_no: int | None = None
) -> int:
    conn = get_connection()
    with conn:
        exp = conn.execute(
            "SELECT id FROM experiments WHERE key=?", (experiment_key,)
        ).fetchone()
        if not exp:
            raise KeyError(f"Experiment not found: {experiment_key}")
        run_id = None
        if run_no is not None:
            r = conn.execute(
                "SELECT id FROM experiment_runs WHERE experiment_id=? AND run_no=?",
                (int(exp[0]), int(run_no)),
            ).fetchone()
            if not r:
                raise KeyError(f"Run not found: {experiment_key} run {run_no}")
            run_id = int(r[0])
        conn.execute(
            "INSERT INTO experiment_artifacts(experiment_id,run_id,type,path,created_at) VALUES (?,?,?,?,?)",
            (int(exp[0]), run_id, type_, path, _utc_now_iso()),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


def save_summary(entity_type: str, entity_id: int, style: str, text: str) -> int:
    if style == "short" and len(text) > 280:
        raise ValueError("Short summaries must be <= 280 characters")
    conn = get_connection()
    with conn:
        conn.execute(
            "INSERT INTO summaries(entity_type,entity_id,style,text,created_at) VALUES (?,?,?,?,?)",
            (entity_type, int(entity_id), style, text, _utc_now_iso()),
        )
        return int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])


# Reads/exports
def get_plan_status(key: str | None = None) -> list[dict[str, Any]]:
    # Auto-initialize DB if it doesn't exist
    if not DB_PATH.exists():
        init_db()
    try:
        conn = get_connection(True) if DB_PATH.exists() else get_connection()
        cur = conn.cursor()
        if key:
            rows = cur.execute(
                """
				SELECT p.key, p.title, p.status, COALESCE(COUNT(t.id),0) AS open_tasks
				FROM feature_plans p
				LEFT JOIN plan_tasks t ON t.plan_id=p.id AND t.status!='completed'
				WHERE p.key=?
				GROUP BY p.id
				""",
                (key,),
            ).fetchall()
        else:
            rows = cur.execute(
                """
				SELECT p.key, p.title, p.status, COALESCE(COUNT(t.id),0) AS open_tasks
				FROM feature_plans p
				LEFT JOIN plan_tasks t ON t.plan_id=p.id AND t.status!='completed'
				GROUP BY p.id
				ORDER BY p.updated_at DESC
				"""
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        # If tables don't exist, initialize and return empty list
        init_db()
        return []


def get_experiment_runs_export() -> list[dict[str, Any]]:
    # Auto-initialize DB if it doesn't exist
    if not DB_PATH.exists():
        init_db()
    try:
        conn = get_connection(True) if DB_PATH.exists() else get_connection()
        rows = conn.execute(
            """
			SELECT e.key AS experiment_key, r.run_no, r.params_json, r.metrics_json, r.outcome, r.created_at
			FROM experiment_runs r
			JOIN experiments e ON e.id = r.experiment_id
			ORDER BY e.key, r.run_no
			"""
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        # If tables don't exist, initialize and return empty list
        init_db()
        return []


__all__ = [
    "DB_PATH",
    "add_debug_note",
    "add_experiment_run",
    "add_plan_task",
    "create_debug_session",
    "get_connection",
    "get_experiment_runs_export",
    "get_plan_status",
    "init_db",
    "link_experiment_artifact",
    "save_summary",
    "set_debug_status",
    "set_plan_status",
    "set_refactor_status",
    "set_task_done",
    "upsert_experiment",
    "upsert_feature_plan",
    "upsert_refactor",
]
