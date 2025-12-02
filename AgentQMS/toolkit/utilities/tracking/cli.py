#!/usr/bin/env python3
# @tool: description=CLI for development/debug tracking and experiment management
# @tool: usage=python AgentQMS/agent_tools/utilities/tracking/cli.py plan status --concise
# @tool: tags=cli,tracking,sqlite
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .db import (
    DB_PATH,
    add_debug_note,
    add_experiment_run,
    add_plan_task,
    get_experiment_runs_export,
    get_plan_status,
    init_db,
    link_experiment_artifact,
    save_summary,
    set_plan_status,
    set_refactor_status,
    set_task_done,
    upsert_experiment,
    upsert_feature_plan,
    upsert_refactor,
)


def _print(obj) -> None:
    if isinstance(obj, dict | list):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(str(obj))


def _create_artifact(artifact_type: str, name: str, title: str) -> str:
    root = Path(__file__).resolve().parents[2]  # AgentQMS/agent_tools
    cmd = [
        sys.executable,
        str(root / "core" / "artifact_workflow.py"),
        "create",
        "--type",
        artifact_type,
        "--name",
        name,
        "--title",
        title,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"artifact creation failed: {res.stderr or res.stdout}")
    for line in (res.stdout or "").splitlines():
        if "Created artifact:" in line:
            return line.split(":", 1)[1].strip()
    return ""


def cmd_init(_: argparse.Namespace) -> None:
    init_db()
    print(f"Initialized DB at {DB_PATH}")


# Plans
def cmd_plan_new(ns: argparse.Namespace) -> None:
    key = ns.key or ns.title.lower().replace(" ", "-")
    pid = upsert_feature_plan(key, ns.title, ns.owner)
    _print({"key": key, "id": pid})


def cmd_plan_status(ns: argparse.Namespace) -> None:
    rows = get_plan_status(ns.key)
    if ns.concise:
        for r in rows:
            print(
                f"{r['key']}: {r['status']} · open_tasks={r['open_tasks']} · {r['title']}"
            )
    else:
        _print(rows)


def cmd_plan_start(ns: argparse.Namespace) -> None:
    set_plan_status(ns.key, "in_progress")
    print("ok")


def cmd_plan_pause(ns: argparse.Namespace) -> None:
    set_plan_status(ns.key, "paused")
    print("ok")


def cmd_plan_done(ns: argparse.Namespace) -> None:
    set_plan_status(ns.key, "completed")
    print("ok")


def cmd_plan_task_add(ns: argparse.Namespace) -> None:
    id_ = add_plan_task(ns.plan_key, ns.title)
    _print({"task_id": id_})


def cmd_plan_task_done(ns: argparse.Namespace) -> None:
    set_task_done(int(ns.task_id))
    print("ok")


# Refactors
def cmd_refactor_new(ns: argparse.Namespace) -> None:
    key = ns.key or ns.title.lower().replace(" ", "-")
    id_ = upsert_refactor(key, ns.title, ns.notes)
    _print({"key": key, "id": id_})


def cmd_refactor_status(_: argparse.Namespace) -> None:
    print("Use start/pause/done to manage status; listing not implemented yet.")


def cmd_refactor_start(ns: argparse.Namespace) -> None:
    set_refactor_status(ns.key, "in_progress")
    print("ok")


def cmd_refactor_pause(ns: argparse.Namespace) -> None:
    set_refactor_status(ns.key, "paused")
    print("ok")


def cmd_refactor_done(ns: argparse.Namespace) -> None:
    set_refactor_status(ns.key, "completed")
    print("ok")


# Debugging
def cmd_debug_new(ns: argparse.Namespace) -> None:
    from .db import create_debug_session

    key = ns.key or ns.title.lower().replace(" ", "-")
    id_ = create_debug_session(key, ns.title, ns.hypothesis or "", ns.scope or "")
    try:
        artifact_path = _create_artifact(
            "bug_report", name=key, title=f"Debug {key} session"
        )
    except Exception:
        artifact_path = ""
    _print({"key": key, "id": id_, "artifact": artifact_path})


def cmd_debug_note(ns: argparse.Namespace) -> None:
    id_ = add_debug_note(ns.key, ns.text)
    _print({"note_id": id_})


def cmd_debug_status(_: argparse.Namespace) -> None:
    print("Debug status listing not implemented yet; will be added with chat-helper.")


# Experiments
def cmd_exp_new(ns: argparse.Namespace) -> None:
    key = ns.key or ns.title.lower().replace(" ", "-")
    id_ = upsert_experiment(key, ns.title, ns.objective or "", ns.owner)
    _print({"key": key, "id": id_})


def cmd_exp_run_add(ns: argparse.Namespace) -> None:
    params = json.loads(ns.params)
    metrics = json.loads(ns.metrics) if ns.metrics else None
    id_ = add_experiment_run(ns.key, int(ns.run_no), params, metrics, ns.outcome)
    _print({"run_id": id_})


def cmd_exp_link(ns: argparse.Namespace) -> None:
    id_ = link_experiment_artifact(ns.key, ns.type, ns.path, ns.run_no)
    _print({"artifact_id": id_})


def _generate_short_summary_text(points: list[str]) -> str:
    joined = "; ".join(p.strip() for p in points if p.strip())
    return joined[:280]


def cmd_exp_summarize(ns: argparse.Namespace) -> None:
    if ns.style not in ("short", "delta"):
        raise SystemExit("--style must be short|delta")
    text = _generate_short_summary_text(ns.points or [])
    from .db import get_connection

    conn = get_connection()
    row = conn.execute("SELECT id FROM experiments WHERE key=?", (ns.key,)).fetchone()
    if not row:
        raise SystemExit(f"Experiment not found: {ns.key}")
    id_ = save_summary("experiment", int(row[0]), ns.style, text)
    artifact_path = _create_artifact(
        "research", name=f"{ns.key}-summary", title=f"Experiment {ns.key} summary"
    )
    if artifact_path:
        link_experiment_artifact(ns.key, "summary", artifact_path, None)
    _print({"summary_id": id_, "artifact": artifact_path})


def cmd_exp_status(ns: argparse.Namespace) -> None:
    from .db import get_connection

    conn = get_connection()
    row = conn.execute(
        "SELECT id,title,status FROM experiments WHERE key=?", (ns.key,)
    ).fetchone()
    if not row:
        raise SystemExit(f"Experiment not found: {ns.key}")
    runs = conn.execute(
        "SELECT run_no,outcome,created_at FROM experiment_runs WHERE experiment_id=? ORDER BY run_no",
        (int(row[0]),),
    ).fetchall()
    print(f"{ns.key}: {row['status']} · {row['title']}")
    for r in runs:
        print(f"  run {r['run_no']}: {r['outcome']} @ {r['created_at']}")


def cmd_exp_export_runs(ns: argparse.Namespace) -> None:
    rows = get_experiment_runs_export()
    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "experiment_key",
                "run_no",
                "params_json",
                "metrics_json",
                "outcome",
                "created_at",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(str(out))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tracking", description="Development/Debug and Experiment tracking CLI"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="Initialize SQLite DB schema")
    init_p.set_defaults(func=cmd_init)

    plans = sub.add_parser("plan", help="Feature plans management")
    psub = plans.add_subparsers(dest="sub", required=True)

    pp = psub.add_parser("new", help="Create or update a plan")
    pp.add_argument("--title", required=True)
    pp.add_argument("--owner", required=False)
    pp.add_argument("--key", required=False)
    pp.set_defaults(func=cmd_plan_new)

    pps = psub.add_parser("status", help="Show plans status")
    pps.add_argument("key", nargs="?")
    pps.add_argument("--concise", action="store_true")
    pps.set_defaults(func=cmd_plan_status)

    pps = psub.add_parser("start", help="Mark a plan in progress")
    pps.add_argument("key")
    pps.set_defaults(func=cmd_plan_start)

    pps = psub.add_parser("pause", help="Pause a plan")
    pps.add_argument("key")
    pps.set_defaults(func=cmd_plan_pause)

    pps = psub.add_parser("done", help="Complete a plan")
    pps.add_argument("key")
    pps.set_defaults(func=cmd_plan_done)

    pt = psub.add_parser("task-add", help="Add a plan task")
    pt.add_argument("plan_key")
    pt.add_argument("--title", required=True)
    pt.set_defaults(func=cmd_plan_task_add)

    ptd = psub.add_parser("task-done", help="Complete a plan task")
    ptd.add_argument("task_id")
    ptd.set_defaults(func=cmd_plan_task_done)

    ref = sub.add_parser("refactor", help="Refactor tracking")
    rsub = ref.add_subparsers(dest="sub", required=True)

    rn = rsub.add_parser("new", help="Create or update a refactor")
    rn.add_argument("--title", required=True)
    rn.add_argument("--notes", required=False)
    rn.add_argument("--key", required=False)
    rn.set_defaults(func=cmd_refactor_new)

    rs = rsub.add_parser("status", help="Show refactor status (TBD)")
    rs.add_argument("key", nargs="?")
    rs.set_defaults(func=cmd_refactor_status)

    rs = rsub.add_parser("start", help="Start refactor")
    rs.add_argument("key")
    rs.set_defaults(func=cmd_refactor_start)

    rs = rsub.add_parser("pause", help="Pause refactor")
    rs.add_argument("key")
    rs.set_defaults(func=cmd_refactor_pause)

    rs = rsub.add_parser("done", help="Complete refactor")
    rs.add_argument("key")
    rs.set_defaults(func=cmd_refactor_done)

    dbg = sub.add_parser("debug", help="Debug sessions")
    dsub = dbg.add_subparsers(dest="sub", required=True)

    dn = dsub.add_parser("new", help="Start a debug session")
    dn.add_argument("--title", required=True)
    dn.add_argument("--hypothesis", required=False)
    dn.add_argument("--scope", required=False)
    dn.add_argument("--key", required=False)
    dn.set_defaults(func=cmd_debug_new)

    dnote = dsub.add_parser("note", help="Add a debug note")
    dnote.add_argument("key")
    dnote.add_argument("--text", required=True)
    dnote.set_defaults(func=cmd_debug_note)

    ds = dsub.add_parser("status", help="Show debug status (TBD)")
    ds.add_argument("key", nargs="?")
    ds.set_defaults(func=cmd_debug_status)

    exp = sub.add_parser("exp", help="Experiments")
    esub = exp.add_subparsers(dest="sub", required=True)

    en = esub.add_parser("new", help="Create or update an experiment")
    en.add_argument("--title", required=True)
    en.add_argument("--objective", required=False)
    en.add_argument("--owner", required=False)
    en.add_argument("--key", required=False)
    en.set_defaults(func=cmd_exp_new)

    er = esub.add_parser("run-add", help="Add or replace a run")
    er.add_argument("key")
    er.add_argument("run_no")
    er.add_argument("--params", required=True)
    er.add_argument("--metrics", required=False)
    er.add_argument(
        "--outcome", required=True, choices=["pass", "fail", "inconclusive"]
    )
    er.set_defaults(func=cmd_exp_run_add)

    el = esub.add_parser("link", help="Link an artifact to an experiment")
    el.add_argument("key")
    el.add_argument("--type", required=True)
    el.add_argument("--path", required=True)
    el.add_argument("--run_no", type=int, required=False)
    el.set_defaults(func=cmd_exp_link)

    es = esub.add_parser("summarize", help="Save a concise summary for an experiment")
    es.add_argument("key")
    es.add_argument("--style", required=True, choices=["short", "delta"])
    es.add_argument("--points", action="append", default=[])
    es.set_defaults(func=cmd_exp_summarize)

    ets = esub.add_parser("status", help="Show experiment status")
    ets.add_argument("key")
    ets.set_defaults(func=cmd_exp_status)

    eex = esub.add_parser("export-runs", help="Export experiment runs to CSV")
    eex.add_argument("--out", required=True)
    eex.set_defaults(func=cmd_exp_export_runs)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    try:
        ns.func(ns)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
