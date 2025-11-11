#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = REPO_ROOT / "docs" / "artifacts"
ARCHIVE_DIR = ARTIFACTS_DIR / "_archived"
VERSION_FILE = REPO_ROOT / "project_version.yaml"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("\n---\n", 1)
    if len(parts) != 2:
        return {}, text
    fm = yaml.safe_load(parts[0].split("---\n", 1)[1]) or {}
    return fm, parts[1]


def render_frontmatter(fm: dict) -> str:
    return "---\n" + yaml.safe_dump(fm, sort_keys=False).strip() + "\n---\n"


def version_tuple(v: str) -> tuple[int, int, int]:
    parts = (v or "0.0.0").split(".")
    parts += ["0"] * (3 - len(parts))
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def should_deprecate(fm: dict, current_version: str) -> bool:
    cur = version_tuple(current_version)
    deprecated_after = fm.get("deprecated_after")
    max_version = fm.get("max_version")
    if deprecated_after and version_tuple(str(deprecated_after)) <= cur:
        return True
    return bool(max_version and version_tuple(str(max_version)) < cur)


def should_hide(fm: dict, current_version: str) -> bool:
    """Check if doc should be hidden (min_version not reached yet)."""
    cur = version_tuple(current_version)
    min_version = fm.get("min_version")
    return bool(min_version and version_tuple(str(min_version)) > cur)


def is_internal(fm: dict) -> bool:
    """Check if doc has internal tag (should be suppressed in public indices)."""
    tags = fm.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    return "internal" in tags or "Internal" in tags


def process_file(
    md_path: Path, current_version: str, dry_run: bool
) -> tuple[bool, str, dict]:
    text = md_path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)
    if not fm:
        return False, "no-frontmatter", {}

    # Check if should be hidden (min_version not reached)
    if should_hide(fm, current_version):
        return (
            False,
            "hidden",
            {"reason": "min_version", "min_version": fm.get("min_version")},
        )

    if not should_deprecate(fm, current_version):
        return False, "skip", {}

    # Update status and add notice
    fm["status"] = "deprecated"
    notice_lines = [
        ":warning: Deprecated due to project version update.",
        f"- Deprecated as of version {current_version}.",
    ]
    if fm.get("superseded_by"):
        notice_lines.append(f"- Superseded by: {fm['superseded_by']}")
    notice = "\n" + "\n".join(notice_lines) + "\n\n"

    new_text = render_frontmatter(fm) + notice + body

    if dry_run:
        return True, "would-deprecate", {}

    # Write back and move to archive mirror path
    md_path.write_text(new_text, encoding="utf-8")
    rel = md_path.relative_to(ARTIFACTS_DIR)
    target = ARCHIVE_DIR / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(md_path), str(target))
    return True, "deprecated", {}


def get_docs_health(current_version: str) -> dict:
    """Get health status of all docs for dashboard."""
    health = {
        "total": 0,
        "active": 0,
        "deprecated": 0,
        "hidden": 0,
        "internal": 0,
        "pending_deprecation": [],
        "hidden_docs": [],
    }
    for md_path in ARTIFACTS_DIR.rglob("*.md"):
        if "_archived" in md_path.parts:
            continue
        health["total"] += 1
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        fm, _ = parse_frontmatter(text)
        if not fm:
            continue
        status = fm.get("status", "active")
        if status == "deprecated":
            health["deprecated"] += 1
        else:
            health["active"] += 1
            if should_deprecate(fm, current_version):
                health["pending_deprecation"].append(
                    {
                        "path": str(md_path.relative_to(REPO_ROOT)),
                        "title": fm.get("title", "Untitled"),
                        "deprecated_after": fm.get("deprecated_after"),
                        "max_version": fm.get("max_version"),
                    }
                )
            if should_hide(fm, current_version):
                health["hidden"] += 1
                health["hidden_docs"].append(
                    {
                        "path": str(md_path.relative_to(REPO_ROOT)),
                        "title": fm.get("title", "Untitled"),
                        "min_version": fm.get("min_version"),
                    }
                )
            if is_internal(fm):
                health["internal"] += 1
    return health


def main() -> int:
    p = argparse.ArgumentParser(
        description="Deprecate docs based on project_version.yaml and frontmatter policy"
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--health", action="store_true", help="Output health status as JSON")
    args = p.parse_args()

    if not VERSION_FILE.exists():
        print(f"Missing version file: {VERSION_FILE}")
        return 2
    version_info = yaml.safe_load(VERSION_FILE.read_text(encoding="utf-8")) or {}
    current_version = str(version_info.get("version", "0.0.0"))

    if args.health:
        import json

        health = get_docs_health(current_version)
        print(json.dumps(health, indent=2))
        return 0

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    changed = 0
    scanned = 0
    hidden_count = 0
    for md_path in ARTIFACTS_DIR.rglob("*.md"):
        if "_archived" in md_path.parts:
            continue
        scanned += 1
        did, status, meta = process_file(md_path, current_version, args.dry_run)
        if did:
            changed += 1
        if status == "hidden":
            hidden_count += 1
        print(f"{md_path}: {status}")

    print(f"Scanned: {scanned}, Deprecated: {changed}, Hidden: {hidden_count}")
    # Exit non-zero on dry-run with changes to help CI detect pending deprecations
    if args.dry_run and changed > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
