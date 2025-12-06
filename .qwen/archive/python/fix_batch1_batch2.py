#!/usr/bin/env python3
"""
Batch frontmatter fixer for artifacts with Qwen-generated issues.
Quickly fixes the 12 batch 1 & 2 files to have correct frontmatter.
"""

from datetime import datetime
from pathlib import Path

# Hardcoded list of batch 1 & 2 assessment artifacts
BATCH1_ASSESSMENTS = [
    "docs/artifacts/assessments/2025-11-11_2343_assessment-ai-documentation-and-scripts-cleanup.md",
    "docs/artifacts/assessments/2025-11-12_1401_assessment-merge-complete-summary---main-and-streamlit-branches.md",
    "docs/artifacts/assessments/2025-11-12_1419_assessment-scripts-directory-audit-and-reorganization.md",
    "docs/artifacts/assessments/2025-11-12_1451_assessment-documentation-and-artifact-architecture.md",
    "docs/artifacts/assessments/2025-11-12_1520_assessment-ui-directory-legacy-files-audit.md",
    "docs/artifacts/assessments/2025-11-12_1547_assessment-data-contracts-update-feasibility.md",
    "docs/artifacts/assessments/2025-11-12_1200_assessment-data-contract-assessment.md",
    "docs/artifacts/assessments/2025-11-12_1200_assessment-refactoring-assessment.md",
    "docs/artifacts/assessments/2025-11-16_1654_assessment-ai-collaboration-framework-extraction-and-standardization.md",
    "docs/artifacts/assessments/2025-11-17_0114_assessment-streamlit-command-builder-performance---page-switch-delays.md",
]

# Batch 2 assessments
BATCH2_ASSESSMENTS = [
    "docs/artifacts/assessments/2025-11-17_0136_assessment-unified-ocr-app-performance.md",
    "docs/artifacts/assessments/2025-11-18_0109_assessment-albumentations-playground-architecture.md",
    "docs/artifacts/assessments/2025-11-19_1939_assessment-upstage-document-parsing-playground-tech-stack.md",
    "docs/artifacts/assessments/2025-11-20_1227_assessment-ai-documentation-entry-points-audit-and-reorganization-plan.md",
    "docs/artifacts/assessments/2025-11-20_1333_assessment-chakra-ui-v3-remaining-build-errors.md",
    "docs/artifacts/assessments/2025-11-21_0231_assessment-text-recognition-implementation-feasibility.md",
    "docs/artifacts/assessments/2025-11-22_0000_assessment-project-composition.md",
    "docs/artifacts/assessments/2025-12-05_2142_assessment-train-py-refactoring.md",
]

# Batch 2 bug reports
BATCH2_BUG_REPORTS = [
    "docs/artifacts/bug_reports/2025-11-28_0000_BUG_001_dominant-edge-extension-failure.md",
    "docs/artifacts/bug_reports/2025-12-03_1100_BUG_001_inference-studio-offsets-data-contract.md",
    "docs/artifacts/bug_reports/2025-12-03_2300_BUG_002_inference-studio-visual-padding-mismatch.md",
]


def extract_timestamp_and_name(filename: str) -> tuple[str, str]:
    """Extract timestamp and name parts from filename."""
    parts = filename.split("_", 3)
    if len(parts) >= 4:
        timestamp = f"{parts[0]}_{parts[1]}"  # YYYY-MM-DD_HHMM
        name = "_".join(parts[3:]).replace(".md", "").replace("-", " ").title()
        return timestamp, name
    return "", filename.replace(".md", "").title()


def normalize_date(date_str: str) -> str:
    """Convert various date formats to 'YYYY-MM-DD HH:MM (KST)'."""
    date_str = date_str.strip()

    # Already correct
    if " (KST)" in date_str:
        return date_str

    # Parse date and time separately
    if "_" in date_str:
        # Format: 2025-11-11_2343
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d_%H%M")
            return dt.strftime("%Y-%m-%d %H:%M (KST)")
        except ValueError:
            pass

    # Just date
    try:
        dt = datetime.strptime(date_str.split()[0], "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d 12:00 (KST)")
    except ValueError:
        pass

    # Fallback
    return datetime.now().strftime("%Y-%m-%d 12:00 (KST)")


def build_assessment_frontmatter(title: str, date_str: str) -> str:
    """Build correct assessment frontmatter."""
    normalized_date = normalize_date(date_str)
    return f"""---
type: assessment
title: "{title}"
date: "{normalized_date}"
category: evaluation
status: active
version: "1.0"
tags:
  - assessment
author: ai-agent
branch: main
---"""


def build_bug_report_frontmatter(title: str, date_str: str) -> str:
    """Build correct bug report frontmatter."""
    normalized_date = normalize_date(date_str)
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
            # Extract quoted title
            title = line.split(":", 1)[1].strip().strip('"\'')
            return title
    return ""


def fix_file(file_path: str, is_assessment: bool = True) -> bool:
    """Fix frontmatter in a single file."""
    path = Path(file_path)

    if not path.exists():
        print(f"⚠️  {path.name}: File not found")
        return False

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ {path.name}: Error reading - {e}")
        return False

    # Extract body (everything after second ---)
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            body = content[end + 3:].lstrip("\n")
        else:
            body = content
    else:
        body = content

    # Try to extract title from existing frontmatter
    title = extract_title_from_frontmatter(content)

    # Fall back to filename if no title found
    if not title:
        timestamp, base_name = extract_timestamp_and_name(path.name)
        title = base_name.replace("Assessment", "").replace("Bug", "").strip()
        if not title:
            title = base_name

    # Extract timestamp from filename for date
    timestamp = path.name.split("_")[0] + "_" + path.name.split("_")[1]

    # Build new frontmatter
    if is_assessment:
        frontmatter = build_assessment_frontmatter(title, timestamp)
    else:
        frontmatter = build_bug_report_frontmatter(title, timestamp)

    # Write new content
    new_content = frontmatter + "\n\n" + body
    path.write_text(new_content, encoding="utf-8")
    print(f"✅ {path.name}: Fixed")
    return True


def main():
    """Fix all batch 1 & 2 artifacts."""
    print("🔧 Fixing batch 1 & 2 artifact frontmatter...\n")

    all_files = BATCH1_ASSESSMENTS + BATCH2_ASSESSMENTS + BATCH2_BUG_REPORTS
    fixed = 0

    for file_path in all_files:
        is_assessment = "assessment" in file_path
        if fix_file(file_path, is_assessment=is_assessment):
            fixed += 1

    print(f"\n📊 Fixed {fixed}/{len(all_files)} files")
    print("✨ Run validation: python AgentQMS/agent_tools/compliance/validate_artifacts.py --all")


if __name__ == "__main__":
    main()
