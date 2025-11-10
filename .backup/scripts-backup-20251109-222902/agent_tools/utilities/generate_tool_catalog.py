#!/usr/bin/env python3
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Calculate paths: __file__ is in utilities/, so parents[1] = scripts/agent_tools
ROOT = Path(__file__).resolve().parents[1]  # scripts/agent_tools
PROJECT_ROOT = (
    ROOT.parent.parent
)  # project root (scripts/agent_tools -> scripts -> root)


@dataclass
class ToolMeta:
    path: Path
    category: str
    name: str
    description: str
    is_cli: bool
    usage_hint: str
    tags: list[str]


EXCLUDE_DIRS = {"__pycache__", "_deprecated"}
CATEGORIES = {"core", "documentation", "compliance", "utilities", "maintenance"}


def discover_python_files(base: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(base):
        rel = Path(dirpath).relative_to(base)
        parts = set(rel.parts)
        if parts & EXCLUDE_DIRS:
            continue
        for fn in filenames:
            if fn.endswith(".py") and not fn.startswith("_"):
                yield Path(dirpath) / fn


def infer_category(p: Path) -> str:
    # category is immediate child under scripts/agent_tools
    parts = p.relative_to(ROOT).parts
    if parts:
        top = parts[0]
        return top if top in CATEGORIES else "utilities"
    return "utilities"


def parse_metadata_block(src: str) -> tuple[str | None, str | None, list[str]]:
    """Parse metadata from top-of-file comments.

    Format:
    # @tool: description=My tool description
    # @tool: usage=python script.py --arg value
    # @tool: tags=cli,validation

    Or simpler:
    # Tool: My tool description
    # Usage: python script.py --arg value
    # Tags: cli, validation

    Returns (description, usage_hint, tags).
    """
    lines = src.split("\n")[:30]  # check first 30 lines only
    desc = None
    usage = None
    tags: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            break  # stop at first non-comment line

        # @tool: format
        if "@tool:" in stripped:
            if "description=" in stripped:
                desc = stripped.split("description=", 1)[1].strip()
            elif "usage=" in stripped:
                usage = stripped.split("usage=", 1)[1].strip()
            elif "tags=" in stripped:
                tag_str = stripped.split("tags=", 1)[1].strip()
                tags.extend(t.strip() for t in tag_str.split(",") if t.strip())
        # Simple format
        elif stripped.startswith("# Tool:"):
            desc = stripped[7:].strip()
        elif stripped.startswith("# Usage:"):
            usage = stripped[8:].strip()
        elif stripped.startswith("# Tags:"):
            tag_str = stripped[7:].strip()
            tags.extend(t.strip() for t in tag_str.split(",") if t.strip())

    return (desc, usage, tags)


def parse_module(p: Path) -> tuple[str, bool, str, list[str]]:
    """Return (doc, is_cli, usage_hint, tags).

    Heuristics:
    - doc: metadata block description, or module docstring (first line, trimmed to 80 chars)
    - is_cli: argparse import or main() guard
    - usage_hint: metadata block usage, or first argparse.add_argument occurrence line or short hint
    - tags: metadata block tags, or based on directory and simple keywords
    """
    try:
        src = p.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return ("", False, "", [])

    # Try metadata block first
    meta_desc, meta_usage, meta_tags = parse_metadata_block(src)

    # Fallback to docstring
    doc = meta_desc or (ast.get_docstring(tree) or "")
    if "\n" in doc:
        doc = doc.split("\n", 1)[0]
    doc = doc.strip()[:80]

    is_cli = ("argparse" in src) or ('if __name__ == "__main__"' in src)
    usage_hint = meta_usage or ""
    if not usage_hint and "add_argument(" in src:
        usage_hint = "argparse CLI"

    tags: list[str] = []
    if meta_tags:
        tags.extend(meta_tags)
    else:
        # Heuristic-based tags
        low = src.lower()
        if "streamlit" in low:
            tags.append("ui")
        if "sqlite3" in low:
            tags.append("sqlite")
        if "ast.parse" in low:
            tags.append("ast")
        if "requests" in low:
            tags.append("network")

    return (doc, is_cli, usage_hint, tags)


def gather_tools() -> list[ToolMeta]:
    tools: list[ToolMeta] = []
    for f in discover_python_files(ROOT):
        # skip this generator itself
        if f == Path(__file__).resolve():
            continue
        cat = infer_category(f)
        name = f.stem
        doc, is_cli, usage_hint, tags = parse_module(f)
        if not doc:
            doc = f"{name} script"
        tm = ToolMeta(
            path=f,
            category=cat,
            name=name,
            description=doc,
            is_cli=is_cli,
            usage_hint=usage_hint,
            tags=tags,
        )
        tools.append(tm)
    # deterministic order: category, name
    tools.sort(key=lambda t: (t.category, t.name))
    return tools


def write_scripts_index(tools: list[ToolMeta]) -> None:
    out = ROOT / "index.md"
    lines: list[str] = []
    lines.append("# Agent Tools Index (generated)\n")
    cur = None
    for t in tools:
        if t.category != cur:
            cur = t.category
            lines.append(f"\n## {cur}\n")
            lines.append("| Tool | Description | CLI | Tags |\n")
            lines.append("|---|---|---|---|\n")
        cli = "yes" if t.is_cli else "no"
        tags = ", ".join(t.tags) if t.tags else "-"
        rel = t.path.relative_to(ROOT)
        lines.append(f"| `{rel}` | {t.description} | {cli} | {tags} |\n")
    out.write_text("\n".join(lines), encoding="utf-8")


def write_docs_catalog(tools: list[ToolMeta]) -> None:
    out = PROJECT_ROOT / "docs" / "ai_agent" / "automation" / "tool_catalog.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("---")
    lines.append('title: "Agent Tool Catalog (generated)"')
    lines.append('date: ""')
    lines.append('type: "guide"')
    lines.append('category: "ai_agent"')
    lines.append('status: "active"')
    lines.append('version: "1.0"')
    lines.append('tags: ["agent-tools", "catalog"]')
    lines.append("---\n")
    lines.append("Agent Tool Catalog (generated)\n==============================\n")
    cur = None
    for t in tools:
        if t.category != cur:
            cur = t.category
            lines.append(f"\n## {cur}\n")
        rel = t.path.relative_to(ROOT)
        tags = ", ".join(t.tags) if t.tags else "-"
        usage = f" — usage: {t.usage_hint}" if t.usage_hint else ""
        lines.append(
            f"- **{t.name}**: {t.description} (`{rel}`) — CLI: {'yes' if t.is_cli else 'no'}{usage} — tags: {tags}"
        )
    out.write_text("\n".join(lines), encoding="utf-8")


def check_readme_sprawl() -> list[str]:
    """Check for README sprawl and return violations."""
    violations: list[str] = []
    readme_files: list[Path] = []

    # Find all README*.md files under scripts/agent_tools
    for dirpath, dirnames, filenames in os.walk(ROOT):
        for fn in filenames:
            if fn.startswith("README") and fn.endswith(".md"):
                readme_files.append(Path(dirpath) / fn)

    # Check root level: only one README.md allowed
    root_readmes = [
        f for f in readme_files if f.parent == ROOT and f.name == "README.md"
    ]
    if len(root_readmes) > 1:
        violations.append(
            f"Multiple README.md files at root: {[str(f) for f in root_readmes]}"
        )

    # Check for README_*.md at root (should be pointers or removed)
    root_readme_variants = [
        f for f in readme_files if f.parent == ROOT and f.name != "README.md"
    ]
    if root_readme_variants:
        violations.append(
            f"README variants at root (should be pointers or removed): {[str(f) for f in root_readme_variants]}"
        )

    # Check subdirectories: suggest POINTER.md instead of README.md
    subdir_readmes = [f for f in readme_files if f.parent != ROOT]
    if subdir_readmes:
        suggestions = []
        for f in subdir_readmes:
            pointer_path = f.parent / "POINTER.md"
            suggestions.append(
                f"  {f.relative_to(ROOT)} → consider {pointer_path.relative_to(ROOT)} (max 10 lines)"
            )
        violations.append(
            "Subdirectory READMEs found (suggest POINTER.md instead):\n"
            + "\n".join(suggestions)
        )

    return violations


def main() -> int:
    tools = gather_tools()
    write_scripts_index(tools)
    write_docs_catalog(tools)
    print(
        f"Wrote: {ROOT / 'index.md'} and {PROJECT_ROOT / 'docs/ai_agent/automation/tool_catalog.md'}"
    )

    # Check for README sprawl
    violations = check_readme_sprawl()
    if violations:
        print("\n⚠️  README sprawl violations detected:")
        for v in violations:
            print(f"  {v}")
        print(
            "\nPolicy: Only one README.md at root; subdirs should use POINTER.md (max 10 lines)"
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
