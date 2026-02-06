#!/usr/bin/env python3
"""Audit ocr/core/ to identify detection-specific vs truly shared code."""

from pathlib import Path
from collections import defaultdict

# Keywords that indicate detection-specific code
DETECTION_KEYWORDS = [
    'polygon', 'box', 'DBNet', 'CRAFT', 'PAN', 'detection', 'CLEval',
    'binary_map', 'prob_map', 'thresh_map', 'inverse_matrix',
    'get_polygons', 'DetectionHead', 'DetectionLoss', 'text_region',
    'shrink_ratio', 'pyclipper', 'contour', 'dilate', 'erode'
]

# Keywords that indicate truly shared code
SHARED_KEYWORDS = [
    'lightning', 'registry', 'config', 'logging', 'git',
    'BaseModel', 'validator', 'Field', 'cache'
]

def audit_file(path: Path) -> tuple[str, int, dict]:
    """Audit a Python file and return its category.

    Returns:
        (category, score, keyword_counts)
    """
    try:
        content = path.read_text()
    except Exception:
        return "ERROR", 0, {}

    lines = len(content.splitlines())

    # Count keyword occurrences
    detection_count = sum(content.lower().count(kw.lower()) for kw in DETECTION_KEYWORDS)
    shared_count = sum(content.lower().count(kw.lower()) for kw in SHARED_KEYWORDS)

    keyword_counts = {
        'detection': detection_count,
        'shared': shared_count,
        'lines': lines
    }

    # Categorize based on keyword density
    if detection_count >= 3:
        category = "DETECTION"
        score = detection_count
    elif shared_count >= 2:
        category = "SHARED"
        score = shared_count
    elif detection_count > 0:
        category = "MAYBE-DETECTION"
        score = detection_count
    else:
        category = "UNCLEAR"
        score = 0

    return category, score, keyword_counts

def main():
    core_path = Path("ocr/core")

    if not core_path.exists():
        print("ERROR: ocr/core/ not found. Run from project root.")
        return

    # Collect results
    results = defaultdict(list)
    total_lines = defaultdict(int)

    print("=" * 80)
    print("OCR/CORE ARCHITECTURE AUDIT")
    print("=" * 80)
    print()

    for py_file in sorted(core_path.rglob("*.py")):
        if py_file.stem == "__init__":
            continue

        category, score, counts = audit_file(py_file)

        rel_path = py_file.relative_to(core_path.parent)
        results[category].append((rel_path, score, counts))
        total_lines[category] += counts['lines']

    # Print results by category
    categories = ["DETECTION", "MAYBE-DETECTION", "SHARED", "UNCLEAR", "ERROR"]

    for category in categories:
        if not results[category]:
            continue

        print(f"\n{'='*80}")
        print(f"{category} FILES ({len(results[category])} files, {total_lines[category]} lines)")
        print(f"{'='*80}")

        # Sort by score (descending)
        sorted_files = sorted(results[category], key=lambda x: x[1], reverse=True)

        for path, score, counts in sorted_files:
            indicator = "🔴" if category == "DETECTION" else \
                       "🟡" if category == "MAYBE-DETECTION" else \
                       "🟢" if category == "SHARED" else "⚪"

            print(f"{indicator} {path}")
            print(f"   Score: {score}, Lines: {counts['lines']}, "
                  f"Detection kw: {counts['detection']}, Shared kw: {counts['shared']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    detection_files = len(results["DETECTION"]) + len(results["MAYBE-DETECTION"])
    detection_lines = total_lines["DETECTION"] + total_lines["MAYBE-DETECTION"]
    shared_files = len(results["SHARED"])
    shared_lines = total_lines["SHARED"]
    total_files = sum(len(v) for v in results.values())
    all_lines = sum(total_lines.values())

    print(f"Total files audited: {total_files}")
    print(f"Total lines: {all_lines:,}")
    print()
    print(f"🔴 Detection-specific: {detection_files} files ({detection_lines:,} lines, "
          f"{detection_lines/all_lines*100:.1f}%)")
    print(f"🟢 Truly shared: {shared_files} files ({shared_lines:,} lines, "
          f"{shared_lines/all_lines*100:.1f}%)")
    print()

    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    if detection_lines > shared_lines:
        print("⚠️  WARNING: ocr/core/ is MAJORITY detection-specific!")
        print(f"   {detection_lines/all_lines*100:.1f}% of code should move to ocr/domains/detection/")
        print()
        print("URGENT: Refactor needed to achieve true 'Domains First' architecture")
    else:
        print("✅ GOOD: ocr/core/ is majority shared code")

    print()
    print("Next steps:")
    print("1. Review files marked 🔴 DETECTION")
    print("2. Move them to ocr/domains/detection/")
    print("3. Update imports")
    print("4. Re-run this audit to verify")
    print()

    # Save detailed report
    report_path = Path("analysis/architecture-migration-2026-01-21/core_audit_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w") as f:
        for category in categories:
            if not results[category]:
                continue
            f.write(f"\n{'='*80}\n")
            f.write(f"{category}\n")
            f.write(f"{'='*80}\n")
            for path, score, counts in sorted(results[category], key=lambda x: x[1], reverse=True):
                f.write(f"{path}\t{score}\t{counts['lines']}\t{counts['detection']}\t{counts['shared']}\n")

    print(f"Detailed report saved to: {report_path}")

if __name__ == "__main__":
    main()
