#!/usr/bin/env python3
"""
AgentQMS CLI - Unified tool interface for AI agents.

Usage from project root:
    ./scripts/aqms validate
    ./scripts/aqms create assessment my-test "My Test Title"
    ./scripts/aqms compliance
    ./scripts/aqms context "fix config loading"
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path for AgentQMS imports
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from AgentQMS.tools.utils.paths import get_project_root


def run_make(target: str, cwd: Path | None = None, **kwargs) -> int:
    """Run Makefile target with proper working directory.

    Args:
        target: Make target name
        cwd: Working directory (defaults to AgentQMS/bin)
        **kwargs: Additional make variables (NAME=x, TITLE=y)

    Returns:
        Exit code from make command
    """
    if cwd is None:
        cwd = get_project_root() / "AgentQMS" / "bin"

    cmd = ["make", target]
    for key, value in kwargs.items():
        cmd.append(f"{key.upper()}={value}")

    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def cmd_validate(args) -> int:
    """Validate artifacts."""
    if args.file:
        return run_make("validate-file", FILE=args.file)
    return run_make("validate")


def cmd_create(args) -> int:
    """Create a new artifact."""
    type_map = {
        "assessment": "create-assessment",
        "implementation_plan": "create-plan",
        "plan": "create-plan",
        "audit": "create-audit",
        "design": "create-design",
        "bug_report": "create-bug-report",
        "bug": "create-bug-report",
    }

    target = type_map.get(args.type)
    if not target:
        print(f"❌ Unknown artifact type: {args.type}")
        print(f"   Valid types: {', '.join(type_map.keys())}")
        return 1

    return run_make(target, NAME=args.name, TITLE=args.title)


def cmd_compliance(args) -> int:
    """Check compliance status."""
    return run_make("compliance")


def cmd_context(args) -> int:
    """Get or suggest context bundles."""
    if args.task:
        return run_make("context-suggest", TASK=args.task)
    if args.type:
        return run_make(f"context-{args.type}")
    return run_make("context-list")


def cmd_fix(args) -> int:
    """Run autofix pipeline."""
    extra = []
    if args.dry_run:
        extra.append("--dry-run")
    if args.limit:
        extra.append(f"--limit {args.limit}")

    if extra:
        return run_make("fix", ARGS=" ".join(extra))
    return run_make("fix")


def cmd_status(args) -> int:
    """Show system status."""
    return run_make("status")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="aqms",
        description="🤖 AgentQMS CLI - Unified tool interface for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aqms validate                    Validate all artifacts
  aqms validate --file path.md     Validate specific file
  aqms create assessment my-test "My Test"    Create assessment
  aqms compliance                  Check compliance status
  aqms context "fix config"        Suggest context for task
  aqms fix --dry-run               Preview fixes
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate artifacts")
    p_validate.add_argument("--file", "-f", help="Validate specific file")
    p_validate.set_defaults(func=cmd_validate)

    # create
    p_create = subparsers.add_parser("create", help="Create new artifact")
    p_create.add_argument("type", help="Artifact type (assessment, plan, audit, design, bug)")
    p_create.add_argument("name", help="Artifact name (kebab-case)")
    p_create.add_argument("title", help="Artifact title")
    p_create.set_defaults(func=cmd_create)

    # compliance
    p_compliance = subparsers.add_parser("compliance", help="Check compliance status")
    p_compliance.set_defaults(func=cmd_compliance)

    # context
    p_context = subparsers.add_parser("context", help="Get or suggest context bundles")
    p_context.add_argument("task", nargs="?", help="Task description for suggestions")
    p_context.add_argument("--type", "-t", choices=["development", "docs", "debug", "plan"],
                           help="Get specific context type")
    p_context.set_defaults(func=cmd_context)

    # fix
    p_fix = subparsers.add_parser("fix", help="Run autofix pipeline")
    p_fix.add_argument("--dry-run", "-n", action="store_true", help="Preview only")
    p_fix.add_argument("--limit", "-l", type=int, help="Limit number of files")
    p_fix.set_defaults(func=cmd_fix)

    # status
    p_status = subparsers.add_parser("status", help="Show system status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
