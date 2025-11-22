#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> int:
    try:
        completed = subprocess.run(cmd, check=False)
        return completed.returncode
    except FileNotFoundError:
        print(f"Command not found: {' '.join(cmd)}")
        return 127


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="agent_tools",
        description="Unified CLI for documentation and quality tooling",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list", help="List available commands")

    p_links = sub.add_parser("prune-links", help="Prune broken markdown links under docs/")
    p_links.add_argument("--root", default="docs")
    p_links.add_argument("--dry-run", action="store_true")

    sub.add_parser("update-artifact-indexes", help="Update artifact INDEX.md files")
    sub.add_parser("auto-generate-index", help="Generate docs indexes (agents, artifacts)")

    p_sitemap = sub.add_parser("sitemap", help="Generate docs/sitemap.md")
    p_sitemap.add_argument("--root", default="docs")

    args = parser.parse_args()
    here = Path(__file__).resolve().parent

    if args.command == "list" or args.command is None:
        print("Available commands:")
        print("  agent_tools list")
        print("  agent_tools prune-links [--root docs] [--dry-run]")
        print("  agent_tools update-artifact-indexes")
        print("  agent_tools auto-generate-index")
        print("  agent_tools sitemap [--root docs]")
        return 0

    if args.command == "prune-links":
        tool = here / "documentation" / "fix_broken_links.py"
        cmd = [sys.executable, str(tool), args.root]
        if args.dry_run:
            cmd.append("--dry-run")
        return run(cmd)

    if args.command == "update-artifact-indexes":
        tool = here / "documentation" / "update_artifact_indexes.py"
        return run([sys.executable, str(tool)])

    if args.command == "auto-generate-index":
        tool = here / "documentation" / "auto_generate_index.py"
        return run([sys.executable, str(tool)])

    if args.command == "sitemap":
        tool = here / "documentation" / "generate_sitemap.py"
        return run([sys.executable, str(tool), args.root])

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
