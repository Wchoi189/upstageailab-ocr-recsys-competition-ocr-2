#!/usr/bin/env python3
"""Validate AgentQMS framework boundaries.

Ensures framework code lives inside the AgentQMS/ container while project
artifacts and documentation remain outside. Intended for both local and CI
workflows.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from AgentQMS.agent_tools.utils.paths import (
    get_artifacts_dir,
    get_docs_dir,
    get_framework_root,
    get_project_root,
)
from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


@dataclass
class BoundaryViolation:
    path: Path
    message: str
    severity: str = "error"


class BoundaryValidator:
    """Encapsulates boundary validation logic."""

    def __init__(self) -> None:
        self.project_root = get_project_root().resolve()
        self.framework_root = get_framework_root().resolve()
        self.violations: list[BoundaryViolation] = []

    def validate(self) -> list[BoundaryViolation]:
        self.violations.clear()
        self._check_framework_boundary()
        self._check_project_boundary()
        self._check_output_paths()
        return list(self.violations)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _check_framework_boundary(self) -> None:
        forbidden = [
            "artifacts",
            "docs",
            "ai_handbook",
            "README.md",
            ".git",
        ]
        for name in forbidden:
            candidate = self.framework_root / name
            if candidate.exists():
                self.violations.append(
                    BoundaryViolation(
                        candidate,
                        f"Project asset '{name}' must not live inside AgentQMS/",
                    )
                )

    def _check_project_boundary(self) -> None:
        # 'scripts' is intentionally allowed at project root to keep project-specific
        # utilities outside the AgentQMS framework. Only legacy framework dirs are
        # flagged here.
        legacy = ["agent", "agent_tools", "quality_management_framework"]
        for name in legacy:
            candidate = self.project_root / name
            if candidate.exists():
                self.violations.append(
                    BoundaryViolation(
                        candidate,
                        f"Legacy framework directory '{name}/' detected outside AgentQMS/",
                        severity="warning",
                    )
                )

    def _check_output_paths(self) -> None:
        artifacts_dir = get_artifacts_dir().resolve()
        docs_dir = get_docs_dir().resolve()

        if _is_within(artifacts_dir, self.framework_root):
            self.violations.append(
                BoundaryViolation(
                    artifacts_dir,
                    "Artifacts directory configured inside AgentQMS/. Update the paths section in .agentqms/settings.yaml.",
                )
            )

        if _is_within(docs_dir, self.framework_root):
            self.violations.append(
                BoundaryViolation(
                    docs_dir,
                    "Documentation directory configured inside AgentQMS/. Recommended to keep outside.",
                    severity="warning",
                )
            )


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AgentQMS boundaries")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON for tooling integration",
    )
    args = parser.parse_args()

    validator = BoundaryValidator()
    violations = validator.validate()

    if args.json:
        payload = [
            {
                "path": str(v.path),
                "message": v.message,
                "severity": v.severity,
            }
            for v in violations
        ]
        print(json.dumps(payload, indent=2))
    else:
        if not violations:
            print("✅ No boundary violations detected")
        else:
            print("❌ Boundary violations detected:\n")
            for violation in violations:
                prefix = "ERROR" if violation.severity == "error" else "WARN"
                print(f"[{prefix}] {violation.path}\n    {violation.message}\n")

    errors = [v for v in violations if v.severity == "error"]
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
