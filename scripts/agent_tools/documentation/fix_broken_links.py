#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urlparse

RE_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def is_url(target: str) -> bool:
    try:
        parsed = urlparse(target)
        return parsed.scheme in {"http", "https"}
    except Exception:
        return False


def resolve_path(md_file: Path, target: str) -> Path:
    # Strip anchors "file.md#section" and line refs "file.py#L10"
    # Keep fragment separate for existence check
    path_part = target.split("#", 1)[0]
    return (md_file.parent / path_part).resolve()


def prune_broken_links(md_path: Path, dry_run: bool = False) -> int:
    """
    Replace broken markdown links [text](target) with plain text 'text'
    when target is a local file path that does not exist.
    Returns number of replacements made.
    """
    original = md_path.read_text(encoding="utf-8")
    changed = 0
    new_text_parts: list[str] = []
    last_index = 0

    for match in RE_LINK.finditer(original):
        text, target = match.group(1), match.group(2).strip()
        start, end = match.span()

        # Preserve everything before this match
        new_text_parts.append(original[last_index:start])

        # Skip URLs and anchors-only links
        if is_url(target) or target.startswith("#"):
            new_text_parts.append(match.group(0))
        else:
            # Resolve and check existence
            resolved = resolve_path(md_path, target)
            if resolved.exists():
                new_text_parts.append(match.group(0))
            else:
                # Prune link: keep visible text only
                new_text_parts.append(text)
                changed += 1

        last_index = end

    new_text_parts.append(original[last_index:])
    if changed > 0 and not dry_run:
        md_path.write_text("".join(new_text_parts), encoding="utf-8")
    return changed


def main() -> int:
    p = argparse.ArgumentParser(description="Prune broken markdown links under a directory.")
    p.add_argument("root", type=str, help="Root directory to scan (e.g., docs)")
    p.add_argument("--dry-run", action="store_true", help="Only report, do not modify files")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root not found: {root}")
        return 2

    total_files = 0
    total_changes = 0
    for md_path in root.rglob("*.md"):
        total_files += 1
        try:
            changed = prune_broken_links(md_path, dry_run=args.dry_run)
            if changed:
                print(f"[fixed] {md_path} -> {changed} pruned")
                total_changes += changed
        except Exception as e:
            print(f"[error] {md_path}: {e}")

    print(f"Scanned {total_files} files; pruned {total_changes} broken links.")
    # Exit non-zero if changes were needed in dry-run
    if args.dry_run and total_changes > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

