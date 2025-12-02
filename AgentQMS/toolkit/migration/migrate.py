#!/usr/bin/env python3
"""Migration CLI for the framework restructure."""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import json
from dataclasses import dataclass
from typing import List, Optional

from AgentQMS.toolkit.utils.config import load_config
from AgentQMS.toolkit.utils.migration import log_migration_event
from AgentQMS.toolkit.utils.paths import (
    get_config_defaults_dir,
    get_framework_root,
    get_project_root,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_PROJECT_FRAMEWORK = """framework:
  name: "AgentQMS Local"
  container_name: AgentQMS
  validation:
    strict_mode: false
    rules:
      naming: true
      frontmatter: true
      structure: true
  automation:
    pre_commit:
      enabled: false
      validate_artifacts: true
"""

DEFAULT_PROJECT_INTERFACE = """interface:
  agent:
    name: "AI Agent"
    version: "1.0"
    environment: "project"
  workflows:
    auto_validate: true
    auto_update_indexes: true
    auto_compliance_check: true
  logging:
    level: "INFO"
    directory: "logs"
  restrictions:
    human_access: false
    validate_context: true
"""

DEFAULT_PROJECT_PATHS = """paths:
  artifacts: artifacts
  docs: docs
  agent_interface: interface
  implementation: toolkit
  project_conventions: project_conventions
  scripts: scripts
  templates: templates
  config_defaults: AgentQMS/config_defaults
"""

DEFAULT_ENVIRONMENT = """# Default environment overrides applied during CI builds
framework:
  validation:
    strict_mode: true
interface:
  logging:
    level: "WARNING"
"""

DEFAULT_LOCAL_OVERRIDES = """# Developer-specific overrides; not committed in real projects
interface:
  logging:
    level: "DEBUG"
paths:
  artifacts: artifacts/local
"""


@dataclass
class Action:
    kind: str
    description: str
    source: Optional[Path] = None
    target: Optional[Path] = None
    content: Optional[str] = None

    def to_dict(self) -> dict:
        data = {
            "kind": self.kind,
            "description": self.description,
        }
        if self.source:
            data["source"] = str(self.source)
        if self.target:
            data["target"] = str(self.target)
        return data


class MigrationManager:
    """Plans and executes migration actions."""

    def __init__(self) -> None:
        self.project_root = get_project_root().resolve()
        self.framework_root = get_framework_root().resolve()
        self.config_defaults_dir = get_config_defaults_dir().resolve()
        self.project_config_dir = self.project_root / "config"

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan_actions(self) -> List[Action]:
        actions: List[Action] = []
        actions.extend(self._plan_directory_renames())
        actions.extend(self._plan_config_defaults())
        actions.extend(self._plan_project_config())
        return actions

    def _plan_directory_renames(self) -> List[Action]:
        actions: List[Action] = []
        replacements = [
            (self.framework_root / "agent", self.framework_root / "agent_interface", "Rename Agent interface directory"),
            (self.framework_root / "conventions", self.framework_root / "project_conventions", "Rename project conventions directory"),
        ]
        for source, target, description in replacements:
            if source.exists() and not target.exists():
                actions.append(
                    Action("rename", description, source=source, target=target)
                )
        return actions

    def _plan_config_defaults(self) -> List[Action]:
        actions: List[Action] = []
        if not self.config_defaults_dir.exists():
            actions.append(
                Action("mkdir", "Create config_defaults directory", target=self.config_defaults_dir)
            )

        moves = [
            (
                self.framework_root / "config" / "framework.yaml",
                self.config_defaults_dir / "framework.yaml",
                "Move framework defaults into config_defaults",
            ),
        ]

        agent_config_candidates = [
            self.framework_root / "agent_interface" / "config" / "agent_config.yaml",
            self.framework_root / "agent" / "config" / "agent_config.yaml",
        ]
        for candidate in agent_config_candidates:
            if candidate.exists():
                moves.append(
                    (
                        candidate,
                        self.config_defaults_dir / "interface.yaml",
                        "Move interface defaults into config_defaults",
                    )
                )
                break

        tool_mapping_candidates = [
            self.framework_root / "agent_interface" / "config" / "tool_mappings.json",
            self.framework_root / "agent" / "config" / "tool_mappings.json",
        ]
        for candidate in tool_mapping_candidates:
            if candidate.exists():
                moves.append(
                    (
                        candidate,
                        self.config_defaults_dir / "tool_mappings.json",
                        "Move tool mappings into config_defaults",
                    )
                )
                break

        for source, target, description in moves:
            if source.exists() and not target.exists():
                actions.append(Action("move", description, source=source, target=target))

        paths_yaml = self.config_defaults_dir / "paths.yaml"
        if not paths_yaml.exists():
            actions.append(
                Action("write", "Create config_defaults/paths.yaml", target=paths_yaml, content=DEFAULT_PROJECT_PATHS)
            )

        return actions

    def _plan_project_config(self) -> List[Action]:
        actions: List[Action] = []
        if not self.project_config_dir.exists():
            actions.append(
                Action("mkdir", "Create root config directory", target=self.project_config_dir)
            )

        for sub in ("environments", "overrides"):
            directory = self.project_config_dir / sub
            if not directory.exists():
                actions.append(Action("mkdir", f"Create config/{sub} directory", target=directory))

        file_plans = [
            (self.project_config_dir / "framework.yaml", DEFAULT_PROJECT_FRAMEWORK, "Seed config/framework.yaml"),
            (self.project_config_dir / "interface.yaml", DEFAULT_PROJECT_INTERFACE, "Seed config/interface.yaml"),
            (self.project_config_dir / "paths.yaml", DEFAULT_PROJECT_PATHS, "Seed config/paths.yaml"),
            (self.project_config_dir / "environments" / "default.yaml", DEFAULT_ENVIRONMENT, "Seed config/environments/default.yaml"),
            (self.project_config_dir / "overrides" / "local.yaml", DEFAULT_LOCAL_OVERRIDES, "Seed config/overrides/local.yaml"),
        ]

        for path, content, description in file_plans:
            if not path.exists():
                actions.append(Action("write", description, target=path, content=content))

        return actions

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def apply(self, actions: List[Action], dry_run: bool = False) -> None:
        for action in actions:
            log_migration_event(f"{'[DRY RUN] ' if dry_run else ''}{action.description}")

            if dry_run:
                continue

            if action.kind == "rename" and action.source and action.target:
                action.source.rename(action.target)
            elif action.kind == "move" and action.source and action.target:
                action.target.parent.mkdir(parents=True, exist_ok=True)
                action.source.rename(action.target)
            elif action.kind == "mkdir" and action.target:
                action.target.mkdir(parents=True, exist_ok=True)
            elif action.kind == "write" and action.target and action.content is not None:
                action.target.parent.mkdir(parents=True, exist_ok=True)
                action.target.write_text(action.content, encoding="utf-8")

        if not dry_run:
            load_config(force=True)
            log_migration_event("Regenerated .agentqms/effective.yaml via ConfigLoader")

    def discover_state(self) -> dict:
        legacy_dirs = [
            str(path)
            for path in (self.framework_root / "agent", self.framework_root / "conventions")
            if path.exists()
        ]
        state = {
            "framework_root": str(self.framework_root),
            "project_root": str(self.project_root),
            "has_config_defaults": self.config_defaults_dir.exists(),
            "has_project_config": self.project_config_dir.exists(),
            "legacy_directories": legacy_dirs,
        }
        return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentQMS migration helper")
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Print migration state without making modifications",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration actions without applying changes",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply migration actions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write discover results as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manager = MigrationManager()
    actions = manager.plan_actions()

    if args.discover:
        payload = {
            "state": manager.discover_state(),
            "pending_actions": [action.to_dict() for action in actions],
        }
        message = json.dumps(payload, indent=2)
        if args.output:
            args.output.write_text(message, encoding="utf-8")
            print(f"🔍 Wrote discovery report to {args.output}")
        else:
            print(message)
        return 0

    if args.dry_run:
        if actions:
            manager.apply(actions, dry_run=True)
            print("✨ Dry run complete. Review migration.log for details.")
        else:
            print("✅ No migration actions required.")
        return 0

    if args.apply:
        if actions:
            manager.apply(actions, dry_run=False)
            print("🎉 Migration applied successfully.")
        else:
            print("✅ Nothing to migrate. Structure already up to date.")
        return 0

    print("No operation selected. Use --discover, --dry-run, or --apply.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

