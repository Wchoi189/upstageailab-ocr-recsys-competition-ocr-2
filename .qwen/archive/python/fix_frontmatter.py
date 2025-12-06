#!/usr/bin/env python3
"""
Frontmatter Correction Script for Artifacts

Fixes frontmatter issues in batch 1 & 2 artifacts:
- Missing/invalid categories → use smart defaults
- Missing/invalid statuses → use 'active'
- Incorrect date formats → reformat to "YYYY-MM-DD HH:MM (KST)"
- Missing required fields → add with defaults
- Type mismatches with filenames → fix to match

Usage:
    python fix_frontmatter.py [--files <pattern>] [--dry-run] [--verbose]
"""

import re
import sys
from datetime import datetime
from pathlib import Path

# Import AgentQMS utilities
try:
    from AgentQMS.agent_tools.utils.paths import get_artifacts_dir
except ImportError:
    print("Error: AgentQMS utilities not available")
    sys.exit(1)


VALID_CATEGORIES = {
    "development", "architecture", "evaluation", "compliance",
    "code_quality", "reference", "planning", "research",
    "troubleshooting", "governance", "meeting", "security"
}

VALID_STATUSES = {
    "active", "draft", "completed", "archived", "deprecated"
}

TYPE_TO_PREFIX = {
    "assessment": "assessment-",
    "bug_report": "BUG_",
    "implementation_plan": "implementation_plan_",
    "design": "design-",
    "research": "research-",
    "audit": "audit-",
    "template": "template-",
}

PREFIX_TO_TYPE = {v: k for k, v in TYPE_TO_PREFIX.items()}

# Default category suggestions based on type
DEFAULT_CATEGORY_FOR_TYPE = {
    "assessment": "evaluation",
    "bug_report": "troubleshooting",
    "implementation_plan": "planning",
    "design": "architecture",
    "research": "research",
    "audit": "compliance",
    "template": "reference",
}


def detect_type_from_filename(filename: str) -> str | None:
    """Detect artifact type from filename prefix after timestamp."""
    # Remove timestamp and .md extension
    after_timestamp = filename.split("_", 2)[-1] if "_" in filename else ""
    if after_timestamp.endswith(".md"):
        after_timestamp = after_timestamp[:-3]

    # Check all known prefixes
    for prefix, type_name in PREFIX_TO_TYPE.items():
        if after_timestamp.startswith(prefix):
            return type_name

    return None


def parse_frontmatter(fm_text: str) -> dict[str, str]:
    """Parse YAML-like frontmatter into a dictionary."""
    fm = {}
    for line in fm_text.split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"\'')
            fm[key] = val
    return fm


def normalize_date_format(date_str: str) -> str:
    """Normalize date to 'YYYY-MM-DD HH:MM (KST)' format."""
    date_str = date_str.strip()

    # Already correct format
    if re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2} \(KST\)$', date_str):
        return date_str

    # Try various formats
    formats_to_try = [
        ("%Y-%m-%d %H:%M (KST)", None),  # Already normalized but check anyway
        ("%Y-%m-%d_%H%M", " (KST)"),     # 2025-11-17_1336 → add time format
        ("%Y-%m-%d %H:%M", " (KST)"),    # 2025-11-17 13:36 → add timezone
        ("%Y-%m-%d", " 12:00 (KST)"),    # 2025-11-17 → add time
        ("%Y-%m-%dT%H:%M:%SZ", None),    # ISO format → reformat
    ]

    for fmt, suffix in formats_to_try:
        try:
            dt = datetime.strptime(date_str.replace("_", " "), fmt)
            normalized = dt.strftime("%Y-%m-%d %H:%M")
            if suffix:
                normalized += suffix
            else:
                normalized += " (KST)"
            return normalized
        except ValueError:
            continue

    # Fallback: use current date
    return datetime.now().strftime("%Y-%m-%d 12:00 (KST)")


def fix_frontmatter(file_path: Path, dry_run: bool = False, verbose: bool = False) -> bool:
    """
    Fix frontmatter in a single file.

    Args:
        file_path: Path to the markdown file
        dry_run: If True, don't write changes
        verbose: If True, print detailed info

    Returns:
        True if file was changed, False otherwise
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ {file_path.name}: Error reading file: {e}")
        return False

    if not content.startswith("---"):
        print(f"⚠️  {file_path.name}: No frontmatter")
        return False

    # Extract frontmatter
    end = content.find("---", 3)
    if end == -1:
        print(f"❌ {file_path.name}: Malformed frontmatter")
        return False

    fm_text = content[3:end]
    body = content[end + 3:].lstrip("\n")

    # Parse frontmatter
    fm = parse_frontmatter(fm_text)
    original_fm = fm.copy()
    changed = False

    # Detect type from filename
    detected_type = detect_type_from_filename(file_path.name)

    # Fix type if detected and mismatch
    if detected_type:
        if fm.get("type") != detected_type:
            fm["type"] = detected_type
            changed = True
            if verbose:
                print(f"  🔧 Type: {original_fm.get('type', 'MISSING')} → {detected_type}")

    # Fix category
    if fm.get("category") not in VALID_CATEGORIES:
        default_cat = DEFAULT_CATEGORY_FOR_TYPE.get(detected_type, "development")
        old_cat = fm.get("category", "MISSING")
        fm["category"] = default_cat
        changed = True
        if verbose:
            print(f"  🔧 Category: {old_cat} → {default_cat}")

    # Fix status
    if fm.get("status") not in VALID_STATUSES:
        old_status = fm.get("status", "MISSING")
        fm["status"] = "active"
        changed = True
        if verbose:
            print(f"  🔧 Status: {old_status} → active")

    # Fix date format
    date_str = fm.get("date", "").strip()
    if date_str:
        normalized = normalize_date_format(date_str)
        if normalized != date_str:
            fm["date"] = normalized
            changed = True
            if verbose:
                print(f"  🔧 Date: {date_str} → {normalized}")

    # Ensure required fields exist
    required = ["type", "title", "date", "category", "status", "version", "tags", "author", "branch"]
    for field in required:
        if field not in fm:
            if field == "version":
                fm[field] = "1.0"
            elif field == "author":
                fm[field] = "ai-agent"
            elif field == "branch":
                fm[field] = "main"
            elif field == "tags":
                fm[field] = "[]"
            else:
                # Should not reach here for required fields
                fm[field] = ""
            changed = True
            if verbose:
                print(f"  🔧 Added {field}: {fm[field]}")

    if not changed:
        print(f"✅ {file_path.name}: Already correct")
        return False

    if dry_run:
        print(f"📋 {file_path.name}: Would fix (dry-run)")
        return True

    # Rebuild frontmatter with proper YAML formatting
    lines = ["---"]
    for key in required:  # Keep order consistent
        if key in fm:
            val = fm[key]
            if key == "tags":
                # Format tags as YAML list
                if val.startswith("["):
                    lines.append(f"{key}: {val}")
                else:
                    lines.append(f"{key}: []")
            else:
                lines.append(f'{key}: "{val}"')
    lines.append("---")

    new_content = "\n".join(lines) + "\n\n" + body
    file_path.write_text(new_content, encoding="utf-8")
    print(f"✅ {file_path.name}: Fixed")
    return True


def main():
    """Fix all artifacts with frontmatter issues."""
    import argparse

    parser = argparse.ArgumentParser(description="Fix artifact frontmatter issues")
    parser.add_argument("--files", help="File pattern to match (e.g., 2025-11*.md)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without writing")
    parser.add_argument("--verbose", action="store_true", help="Print detailed changes")
    args = parser.parse_args()

    artifacts_dir = get_artifacts_dir()
    print(f"🔍 Scanning {artifacts_dir}...")
    print()

    fixed_count = 0
    total_count = 0

    # Find files to process
    if args.files:
        # Search by pattern
        pattern = args.files
        md_files = list(artifacts_dir.rglob(pattern))
    else:
        # All markdown files except INDEX.md
        md_files = list(artifacts_dir.rglob("*.md"))

    md_files = [f for f in md_files if f.name != "INDEX.md"]
    md_files.sort()

    for md_file in md_files:
        total_count += 1
        if fix_frontmatter(md_file, dry_run=args.dry_run, verbose=args.verbose):
            fixed_count += 1

    print()
    if args.dry_run:
        print(f"📊 Would fix {fixed_count}/{total_count} files (dry-run mode)")
    else:
        print(f"📊 Fixed {fixed_count}/{total_count} files")
        if fixed_count > 0:
            print("✨ Run validation to confirm: python AgentQMS/agent_tools/compliance/validate_artifacts.py --all")


if __name__ == "__main__":
    main()
