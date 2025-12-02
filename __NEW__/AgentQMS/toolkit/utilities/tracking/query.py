from __future__ import annotations

from typing import TYPE_CHECKING

from .db import DB_PATH, get_connection, get_plan_status, init_db

if TYPE_CHECKING:
    from collections.abc import Iterable


def ultra_concise(points: Iterable[str]) -> str:
    joined = "; ".join(p.strip() for p in points if p and p.strip())
    return joined[:280]


def get_status(kind: str, key: str | None = None) -> str:
    # Auto-initialize DB if it doesn't exist
    if not DB_PATH.exists():
        init_db()

    kind = kind.lower()
    if kind == "plan":
        rows = get_plan_status(key)
        if not rows:
            return "No plans found."
        return " | ".join(
            f"{r['key']}:{r['status']} open={r['open_tasks']}" for r in rows
        )

    try:
        conn = get_connection()
        if kind == "experiment":
            if not key:
                rows = conn.execute(
                    "SELECT key,status FROM experiments ORDER BY updated_at DESC"
                ).fetchall()
                if not rows:
                    return "No experiments found."
                return " | ".join(f"{r['key']}:{r['status']}" for r in rows)
            row = conn.execute(
                "SELECT id,status FROM experiments WHERE key=?", (key,)
            ).fetchone()
            if not row:
                return f"Experiment not found: {key}"
            runs = conn.execute(
                "SELECT run_no,outcome FROM experiment_runs WHERE experiment_id=? ORDER BY run_no DESC",
                (int(row[0]),),
            ).fetchall()
            if runs:
                latest = runs[0]
                return f"{key}:{row['status']} latest=run{latest['run_no']}:{latest['outcome']}"
            return f"{key}:{row['status']} no-runs"

        if kind == "debug":
            if key:
                row = conn.execute(
                    "SELECT key,status,title FROM debug_sessions WHERE key=?", (key,)
                ).fetchone()
                if not row:
                    return f"Debug not found: {key}"
                return f"{row['key']}:{row['status']} · {row['title']}"
            rows = conn.execute(
                "SELECT key,status FROM debug_sessions ORDER BY started_at DESC"
            ).fetchall()
            if not rows:
                return "No debug sessions."
            return " | ".join(f"{r['key']}:{r['status']}" for r in rows)

        if kind == "refactor":
            rows = conn.execute(
                "SELECT key,status FROM refactors ORDER BY updated_at DESC"
            ).fetchall()
            if not rows:
                return "No refactors."
            return " | ".join(f"{r['key']}:{r['status']}" for r in rows)

        if kind == "all":
            return ultra_concise(
                [
                    "plans: " + get_status("plan"),
                    "experiments: " + get_status("experiment"),
                    "debug: " + get_status("debug"),
                    "refactors: " + get_status("refactor"),
                ]
            )

        return f"Unknown kind: {kind}"
    except Exception:
        # If tables don't exist, initialize and return empty status
        init_db()
        if kind == "plan":
            return "No plans found."
        elif kind == "experiment":
            return "No experiments found."
        elif kind == "debug":
            return "No debug sessions."
        elif kind == "refactor":
            return "No refactors."
        elif kind == "all":
            return "No data found."
        return f"Unknown kind: {kind}"


__all__ = ["get_status", "ultra_concise"]
