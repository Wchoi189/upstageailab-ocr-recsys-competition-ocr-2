#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

EXCLUDE_DIRS = {"node_modules", ".git", ".venv", "site", "build", "dist", ".idea", ".vscode"}


def make_rel(root: Path, p: Path) -> str:
    return str(p.relative_to(root).as_posix())


def generate_sitemap(root_dir: Path) -> str:
    lines: list[str] = ["# Documentation Sitemap", ""]
    for path in sorted(root_dir.rglob("*")):
        if path.is_dir():
            # Skip excluded directories
            if any(part in EXCLUDE_DIRS for part in path.parts):
                continue
            # Only list top-level directories once
            continue
        if path.suffix.lower() not in {".md", ".mdx"}:
            continue
        rel = make_rel(root_dir, path)
        # Indentation based on depth
        depth = len(Path(rel).parents) - 1
        indent = "  " * depth
        lines.append(f"{indent}- [{rel}]({rel})")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a simple sitemap for docs")
    parser.add_argument("root", type=str, help="Docs root (e.g., docs)")
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    if not root_dir.exists():
        print(f"Root not found: {root_dir}")
        return 2

    content = generate_sitemap(root_dir)
    out_path = root_dir / "sitemap.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

