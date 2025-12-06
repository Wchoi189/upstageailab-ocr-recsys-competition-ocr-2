#!/usr/bin/env python3
"""
Complete batch 1 & 2 artifact fix: rename missing prefixes and fix frontmatter.
Run from project root: python .qwen/final_batch_fix.py
"""

import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/workspaces/upstageailab-ocr-recsys-competition-ocr-2")

# Files that need renaming (add assessment- prefix)
BATCH2_NEED_RENAME = [
    ("docs/artifacts/assessments/2025-11-19_1939_upstage-document-parsing-playground-tech-stack.md",
     "docs/artifacts/assessments/2025-11-19_1939_assessment-upstage-document-parsing-playground-tech-stack.md"),
    ("docs/artifacts/assessments/2025-11-20_1227_ai-documentation-entry-points-audit-and-reorganization-plan.md",
     "docs/artifacts/assessments/2025-11-20_1227_assessment-ai-documentation-entry-points-audit-and-reorganization-plan.md"),
    ("docs/artifacts/assessments/2025-11-20_1333_chakra-ui-v3-remaining-build-errors-assessment.md",
     "docs/artifacts/assessments/2025-11-20_1333_assessment-chakra-ui-v3-remaining-build-errors.md"),
    ("docs/artifacts/assessments/2025-11-21_0231_text-recognition-implementation-feasibility-assessment.md",
     "docs/artifacts/assessments/2025-11-21_0231_assessment-text-recognition-implementation-feasibility.md"),
    ("docs/artifacts/assessments/2025-11-22_0000_project-composition-assessment.md",
     "docs/artifacts/assessments/2025-11-22_0000_assessment-project-composition.md"),
]

# Bug reports that need frontmatter
BUG_REPORTS_NEED_FM = [
    "docs/artifacts/bug_reports/2025-11-28_0000_BUG_001_dominant-edge-extension-failure.md",
    "docs/artifacts/bug_reports/2025-12-03_1100_BUG_001_inference-studio-offsets-data-contract.md",
    "docs/artifacts/bug_reports/2025-12-03_2300_BUG_002_inference-studio-visual-padding-mismatch.md",
]


def rename_file(old_path: str, new_path: str) -> bool:
    """Rename file using git mv."""
    old = PROJECT_ROOT / old_path
    new = PROJECT_ROOT / new_path

    if not old.exists():
        print(f"⚠️  {old.name}: Source not found")
        return False

    try:
        # Use git mv to preserve history
        subprocess.run(
            ["git", "mv", str(old), str(new)],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
        )
        print(f"✅ Renamed: {old.name} → {new.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {old.name}: Rename failed - {e}")
        return False


def extract_title_from_frontmatter(content: str) -> str:
    """Extract title from existing YAML frontmatter."""
    if not content.startswith("---"):
        return ""

    end = content.find("---", 3)
    if end == -1:
        return ""

    fm_text = content[3:end]
    for line in fm_text.split("\n"):
        line = line.strip()
        if line.startswith("title:"):
            title = line.split(":", 1)[1].strip().strip('"\'')
            return title
    return ""


def build_bug_report_frontmatter(title: str, timestamp: str) -> str:
    """Build correct bug report frontmatter."""
    # Parse timestamp: YYYY-MM-DD_HHMM
    parts = timestamp.split("_")
    date_part = parts[0]  # YYYY-MM-DD
    time_part = parts[1] if len(parts) > 1 else "1200"  # HHMM
    hour = time_part[:2]
    minute = time_part[2:4] if len(time_part) >= 4 else "00"

    normalized_date = f"{date_part} {hour}:{minute} (KST)"

    return f"""---
type: bug_report
title: "{title}"
date: "{normalized_date}"
category: troubleshooting
status: active
version: "1.0"
tags:
  - bug
author: ai-agent
branch: main
---"""


def fix_bug_report_frontmatter(file_path: str) -> bool:
    """Add/fix frontmatter in bug report file."""
    path = PROJECT_ROOT / file_path

    if not path.exists():
        print(f"⚠️  {path.name}: File not found")
        return False

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ {path.name}: Error reading - {e}")
        return False

    # Extract body (everything after frontmatter if exists)
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            body = content[end + 3:].lstrip("\n")
        else:
            body = content
    else:
        body = content

    # Extract timestamp and title from filename
    parts = path.name.replace(".md", "").split("_")
    timestamp = f"{parts[0]}_{parts[1]}"  # YYYY-MM-DD_HHMM

    # Try to extract title from body (first heading)
    title = extract_title_from_frontmatter(content)
    if not title:
        # Look for first heading
        for line in body.split("\n"):
            if line.startswith("#"):
                title = line.lstrip("#").strip()
                break
        if not title:
            title = path.name.replace(".md", "").replace("_BUG_", " BUG ").replace("-", " ").title()

    # Build new frontmatter
    frontmatter = build_bug_report_frontmatter(title, timestamp)

    # Write new content
    new_content = frontmatter + "\n\n" + body
    path.write_text(new_content, encoding="utf-8")
    print(f"✅ {path.name}: Added frontmatter")
    return True


def main():
    """Execute all fixes."""
    print("🔧 Executing batch 1 & 2 complete fix...\n")
    print("=" * 60)

    print("\n📝 Step 1: Rename batch 2 assessment files (add assessment- prefix)\n")
    renamed = 0
    for old_path, new_path in BATCH2_NEED_RENAME:
        if rename_file(old_path, new_path):
            renamed += 1

    print(f"\n✅ Renamed: {renamed}/{len(BATCH2_NEED_RENAME)} files\n")
    print("=" * 60)

    print("\n📝 Step 2: Add frontmatter to bug reports\n")
    fixed_fm = 0
    for file_path in BUG_REPORTS_NEED_FM:
        if fix_bug_report_frontmatter(file_path):
            fixed_fm += 1

    print(f"\n✅ Fixed: {fixed_fm}/{len(BUG_REPORTS_NEED_FM)} files\n")
    print("=" * 60)

    print("\n📊 SUMMARY")
    print(f"  ✅ Renamed files: {renamed}")
    print(f"  ✅ Fixed frontmatter: {fixed_fm}")
    print("\n  ✨ Next: Run validation")
    print("  $ python AgentQMS/agent_tools/compliance/validate_artifacts.py --all")


if __name__ == "__main__":
    main()
