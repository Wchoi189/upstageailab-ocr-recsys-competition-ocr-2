#!/usr/bin/env python3
"""Smoke gate for project root resolution behavior.

This script is intentionally strict:
- Every scenario uses assertions for validation.
- Any failing scenario causes a non-zero exit status.
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


@dataclass
class ScenarioResult:
    name: str
    expected: str
    actual: str
    status: str
    detail: str = ""


def _run_python_snippet(code: str, *, cwd: Path, env: dict[str, str]) -> str:
    proc = subprocess.run(
        [PYTHON, "-c", code],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        message = proc.stderr.strip() or proc.stdout.strip() or "snippet failed"
        raise AssertionError(f"snippet failed: {message}")
    output = proc.stdout.strip()
    assert output, "snippet produced no output"
    return output.splitlines()[-1].strip()


def _resolve_entry_points(*, cwd: Path, env: dict[str, str]) -> dict[str, str]:
    py_env = dict(env)
    existing_pp = py_env.get("PYTHONPATH", "")
    py_env["PYTHONPATH"] = f"{REPO_ROOT}:{existing_pp}" if existing_pp else str(REPO_ROOT)

    snippets = {
        "config.ConfigLoader": (
            "from AgentQMS.tools.utils.config.config import ConfigLoader;"
            "print(ConfigLoader().project_root.resolve())"
        ),
        "utils.paths.get_project_root": (
            "from AgentQMS.tools.utils.paths import get_project_root;"
            "print(get_project_root().resolve())"
        ),
        "utils.system.paths.get_project_root": (
            "from AgentQMS.tools.utils.system.paths import get_project_root;"
            "print(get_project_root().resolve())"
        ),
        "mcp_server.find_project_root": (
            "from AgentQMS.mcp_server import find_project_root;"
            "print(find_project_root().resolve())"
        ),
        "cli.project_root": "import AgentQMS.cli as cli; print(Path(cli.project_root).resolve())",
    }

    roots: dict[str, str] = {}
    for name, code in snippets.items():
        roots[name] = _run_python_snippet(
            f"from pathlib import Path; {code}",
            cwd=cwd,
            env=py_env,
        )

    roots["bin/aqms.PROJECT_ROOT"] = _run_python_snippet(
        (
            "import runpy; from pathlib import Path; "
            f"g = runpy.run_path('{(REPO_ROOT / 'AgentQMS' / 'bin' / 'aqms').as_posix()}', run_name='aqms_smoke'); "
            "print(Path(g['PROJECT_ROOT']).resolve())"
        ),
        cwd=cwd,
        env=py_env,
    )

    cli_proc = subprocess.run(
        [PYTHON, "-m", "AgentQMS.cli", "status", "--json"],
        cwd=str(cwd),
        env=py_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if cli_proc.returncode != 0:
        message = cli_proc.stderr.strip() or cli_proc.stdout.strip() or "aqms status failed"
        raise AssertionError(f"aqms status failed: {message}")
    try:
        cli_status = json.loads(cli_proc.stdout.strip() or "{}")
        roots["cli.status.project_root"] = str(Path(cli_status["project_root"]).resolve())
    except Exception as exc:  # pragma: no cover - defensive decode guard
        raise AssertionError(f"unable to parse aqms status output: {exc}") from exc

    return roots


def _scenario_env_override() -> ScenarioResult:
    with tempfile.TemporaryDirectory(prefix="smoke_override_") as tmp:
        expected = str(Path(tmp).resolve())
        env = dict(os.environ)
        env["AGENTQMS_PROJECT_ROOT"] = expected

        roots = _resolve_entry_points(cwd=REPO_ROOT, env=env)
        for name, root in roots.items():
            assert root == expected, f"{name} resolved {root}, expected {expected}"

        return ScenarioResult("Env Override", expected, expected, "PASS")


def _scenario_marker_traversal() -> ScenarioResult:
    with tempfile.TemporaryDirectory(prefix="smoke_marker_") as tmp:
        project_root = Path(tmp).resolve()
        (project_root / ".agentqms").mkdir(parents=True, exist_ok=True)
        nested = project_root / "a" / "b" / "c"
        nested.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env.pop("AGENTQMS_PROJECT_ROOT", None)

        roots = _resolve_entry_points(cwd=nested, env=env)
        expected = str(project_root)
        for name, root in roots.items():
            assert root == expected, f"{name} resolved {root}, expected {expected}"

        return ScenarioResult("Marker Traversal", expected, expected, "PASS")


def _scenario_fallback_to_cwd() -> ScenarioResult:
    with tempfile.TemporaryDirectory(prefix="smoke_fallback_") as tmp:
        expected = str(Path(tmp).resolve())
        env = dict(os.environ)
        env.pop("AGENTQMS_PROJECT_ROOT", None)

        roots = _resolve_entry_points(cwd=Path(tmp), env=env)
        for name, root in roots.items():
            assert root == expected, f"{name} resolved {root}, expected {expected}"

        return ScenarioResult("Fallback to CWD", expected, expected, "PASS")


def _scenario_duality_check() -> ScenarioResult:
    from AgentQMS.tools.utils.config import ConfigLoader as PackageConfigLoader
    from AgentQMS.tools.utils.config.config import ConfigLoader as RootConfigLoader
    from AgentQMS.tools.utils.config import YamlCacheLoader
    from AgentQMS.tools.utils.config.loader import ConfigLoader as LegacyUtilityAlias

    assert RootConfigLoader is not YamlCacheLoader, "Root and YAML/cache loaders must remain distinct"
    assert PackageConfigLoader is RootConfigLoader, "Package ConfigLoader must map to canonical root resolver"
    assert LegacyUtilityAlias is YamlCacheLoader, "loader.ConfigLoader alias must map to YamlCacheLoader"
    assert RootConfigLoader.__module__.endswith(".config"), "Root ConfigLoader module mismatch"
    assert YamlCacheLoader.__module__.endswith(".loader"), "YamlCacheLoader module mismatch"

    actual = (
        f"package={PackageConfigLoader.__module__}, "
        f"root={RootConfigLoader.__module__}, "
        f"yaml_cache={YamlCacheLoader.__module__}"
    )
    expected = "package/root -> ...config, yaml_cache -> ...loader"
    return ScenarioResult("Duality Check", expected, actual, "PASS")


def _scenario_global_install_simulation() -> ScenarioResult:
    from AgentQMS.tools.utils.config.config import ConfigLoader

    with tempfile.TemporaryDirectory(prefix="smoke_project_") as project_tmp, tempfile.TemporaryDirectory(
        prefix="smoke_framework_"
    ) as framework_tmp:
        project_root = Path(project_tmp).resolve()
        framework_root = Path(framework_tmp).resolve()

        settings_dir = project_root / ".agentqms"
        settings_dir.mkdir(parents=True, exist_ok=True)
        settings_path = settings_dir / "settings.yaml"
        settings_path.write_text("paths:\n  artifacts: project_artifacts\n", encoding="utf-8")

        loader = ConfigLoader()
        loader.project_root = project_root
        loader.framework_root = framework_root
        config = loader.load(force=True)

        actual_artifacts = str(config.get("paths", {}).get("artifacts"))
        expected_artifacts = "project_artifacts"
        assert actual_artifacts == expected_artifacts, (
            f"settings must load from project_root. got={actual_artifacts}, expected={expected_artifacts}"
        )

        expected_effective = project_root / ".agentqms" / "effective.yaml"
        assert expected_effective.exists(), f"expected runtime snapshot in project root: {expected_effective}"

        return ScenarioResult(
            "Global Install Simulation",
            str(project_root),
            str(loader.project_root),
            "PASS",
        )


SCENARIOS: dict[str, Callable[[], ScenarioResult]] = {
    "override": _scenario_env_override,
    "traversal": _scenario_marker_traversal,
    "fallback": _scenario_fallback_to_cwd,
    "duality": _scenario_duality_check,
    "global-install": _scenario_global_install_simulation,
}


def _print_results(results: list[ScenarioResult]) -> None:
    print("| Test Scenario | Expected Root | Actual Root | Status |")
    print("| :--- | :--- | :--- | :--- |")
    for row in results:
        print(f"| {row.name} | {row.expected} | {row.actual} | {row.status} |")
        if row.detail:
            print(f"# {row.name}: {row.detail}")


def main(argv: list[str]) -> int:
    scenario = argv[1] if len(argv) > 1 else "all"
    selected = SCENARIOS if scenario == "all" else {scenario: SCENARIOS[scenario]}

    results: list[ScenarioResult] = []
    failures = 0

    for key, runner in selected.items():
        try:
            results.append(runner())
        except AssertionError as exc:
            failures += 1
            label = key.replace("-", " ").title()
            results.append(
                ScenarioResult(
                    name=label,
                    expected="(see scenario)",
                    actual="(assertion failed)",
                    status="FAIL",
                    detail=str(exc),
                )
            )

    _print_results(results)

    if failures:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    valid = {"all", *SCENARIOS.keys()}
    if len(sys.argv) > 1 and sys.argv[1] not in valid:
        print(f"Usage: {Path(__file__).name} [all|override|traversal|fallback|duality|global-install]")
        sys.exit(1)
    raise SystemExit(main(sys.argv))
