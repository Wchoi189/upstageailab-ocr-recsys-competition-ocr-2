#!/usr/bin/env python3
"""Smoke-import integration check for all tools.utils.paths consumers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
TARGET_IMPORT = "AgentQMS.tools.utils.paths"
MODULE_ATTR_CANDIDATES = ("PROJECT_ROOT", "project_root", "projectRoot")


@dataclass
class SmokeResult:
    scenario: str
    modules_checked: int
    import_failures: int
    root_attr_mismatches: int
    status: str
    detail: str = ""


def _module_name_from_file(file_path: Path) -> str:
    rel = file_path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def discover_consumers() -> list[str]:
    modules: list[str] = []
    for file_path in sorted((REPO_ROOT / "AgentQMS").rglob("*.py")):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        if TARGET_IMPORT in text:
            modules.append(_module_name_from_file(file_path))
    return modules


def _run_probe(*, cwd: Path, env: dict[str, str], modules: list[str]) -> dict:
    probe = f"""
import importlib
import json
from pathlib import Path
from AgentQMS.tools.utils.paths import get_project_root

modules = {modules!r}
results = {{}}
for module_name in modules:
    try:
        module = importlib.import_module(module_name)
        attrs = {{}}
        for attr_name in {MODULE_ATTR_CANDIDATES!r}:
            if hasattr(module, attr_name):
                value = getattr(module, attr_name)
                if isinstance(value, (str, Path)):
                    attrs[attr_name] = str(Path(value).resolve())
        results[module_name] = {{"ok": True, "attrs": attrs}}
    except Exception as exc:
        results[module_name] = {{"ok": False, "error": f"{{type(exc).__name__}}: {{exc}}"}}

print(json.dumps({{
    "resolved_root": str(get_project_root().resolve()),
    "modules": results
}}))
"""
    py_env = dict(env)
    existing_pp = py_env.get("PYTHONPATH", "")
    py_env["PYTHONPATH"] = f"{REPO_ROOT}:{existing_pp}" if existing_pp else str(REPO_ROOT)
    proc = subprocess.run(
        [PYTHON, "-c", probe],
        cwd=str(cwd),
        env=py_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(proc.stderr.strip() or proc.stdout.strip() or "probe failed")
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"failed to parse probe output: {exc}") from exc


def _assert_probe(payload: dict, expected_root: Path) -> tuple[int, int]:
    import_failures = 0
    root_attr_mismatches = 0

    actual_root = Path(payload["resolved_root"]).resolve()
    assert actual_root == expected_root.resolve(), f"resolved root mismatch: {actual_root} != {expected_root}"

    for module_name, result in payload["modules"].items():
        if not result["ok"]:
            import_failures += 1
            continue
        for attr_name, value in result["attrs"].items():
            if attr_name.lower().endswith("root") and Path(value).resolve() != expected_root.resolve():
                root_attr_mismatches += 1
                print(
                    f"# root mismatch: {module_name}.{attr_name}={Path(value).resolve()} expected={expected_root}",
                    file=sys.stderr,
                )
    return import_failures, root_attr_mismatches


def scenario_env_override(modules: list[str]) -> SmokeResult:
    with tempfile.TemporaryDirectory(prefix="paths_env_override_") as tmp:
        expected_root = Path(tmp).resolve()
        env = dict(os.environ)
        env["AGENTQMS_PROJECT_ROOT"] = str(expected_root)
        payload = _run_probe(cwd=REPO_ROOT, env=env, modules=modules)
        import_failures, root_mismatches = _assert_probe(payload, expected_root)
        status = "PASS" if import_failures == 0 and root_mismatches == 0 else "FAIL"
        return SmokeResult("Env Override", len(modules), import_failures, root_mismatches, status)


def scenario_marker_traversal(modules: list[str]) -> SmokeResult:
    with tempfile.TemporaryDirectory(prefix="paths_marker_traversal_") as tmp:
        expected_root = Path(tmp).resolve()
        (expected_root / ".agentqms").mkdir(parents=True, exist_ok=True)
        nested = expected_root / "nested" / "deep"
        nested.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.pop("AGENTQMS_PROJECT_ROOT", None)
        payload = _run_probe(cwd=nested, env=env, modules=modules)
        import_failures, root_mismatches = _assert_probe(payload, expected_root)
        status = "PASS" if import_failures == 0 and root_mismatches == 0 else "FAIL"
        return SmokeResult("Marker Traversal", len(modules), import_failures, root_mismatches, status)


def print_results(results: list[SmokeResult]) -> None:
    print("| Scenario | Modules Checked | Import Failures | Root Attr Mismatches | Status |")
    print("| :--- | ---: | ---: | ---: | :--- |")
    for row in results:
        print(
            f"| {row.scenario} | {row.modules_checked} | {row.import_failures} | "
            f"{row.root_attr_mismatches} | {row.status} |"
        )
        if row.detail:
            print(f"# {row.scenario}: {row.detail}")


def main() -> int:
    modules = discover_consumers()
    assert modules, "no paths consumers discovered"

    results: list[SmokeResult] = []
    failures = 0
    for runner in (scenario_env_override, scenario_marker_traversal):
        try:
            results.append(runner(modules))
        except AssertionError as exc:
            failures += 1
            results.append(SmokeResult(runner.__name__, len(modules), 0, 0, "FAIL", str(exc)))

    print_results(results)

    if failures or any(r.status != "PASS" for r in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
