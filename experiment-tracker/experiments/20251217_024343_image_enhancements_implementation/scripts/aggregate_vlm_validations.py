#!/usr/bin/env python3
"""
scripts/aggregate_vlm_validations.py
Aggregate VLM validation reports into summary statistics
"""

import argparse
import re
from pathlib import Path


def extract_score(text: str, pattern: str) -> float | None:
    """Extract numeric score from markdown text."""
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_validation_report(report_path: Path) -> dict:
    """Parse a single VLM validation report."""
    text = report_path.read_text()

    # Extract image ID
    image_id = report_path.stem.replace("_validation", "")

    # Extract metrics
    overall_improvement = extract_score(text, r"Overall Improvement.*?(\d+\.\d+)")
    background_delta = extract_score(text, r"Background.*?Δ\s*=\s*([+-]?\d+\.\d+)")
    alignment_delta = extract_score(text, r"(?:Alignment|Text Orientation).*?Δ\s*=\s*([+-]?\d+\.\d+)")
    contrast_delta = extract_score(text, r"Contrast.*?Δ\s*=\s*([+-]?\d+\.\d+)")

    # Extract success status
    success_match = re.search(r"Overall Success.*?(Pass|Partial|Fail)", text, re.IGNORECASE)
    success = success_match.group(1) if success_match else "Unknown"

    return {
        "image_id": image_id,
        "overall_improvement": overall_improvement,
        "background_delta": background_delta,
        "alignment_delta": alignment_delta,
        "contrast_delta": contrast_delta,
        "success": success,
    }


def aggregate_validations(input_dir: Path, output_file: Path):
    """Aggregate all VLM validation reports into summary."""
    reports = sorted(input_dir.glob("*_validation.md"))

    if not reports:
        print(f"No validation reports found in: {input_dir}")
        return

    # Parse all reports
    results = []
    for report in reports:
        try:
            result = parse_validation_report(report)
            results.append(result)
        except Exception as e:
            print(f"Error parsing {report.name}: {e}")

    if not results:
        print("No valid reports parsed")
        return

    # Generate summary markdown
    with open(output_file, "w") as f:
        f.write("# VLM Validation Summary\n\n")
        f.write(f"**Phase**: {input_dir.name}\n")
        f.write(f"**Total Images**: {len(results)}\n")
        f.write(f"**Date**: {output_file.parent.name}\n\n")

        # Success breakdown
        pass_count = sum(1 for r in results if r["success"].lower() == "pass")
        partial_count = sum(1 for r in results if r["success"].lower() == "partial")
        fail_count = sum(1 for r in results if r["success"].lower() == "fail")

        f.write("## Success Rate\n\n")
        f.write(f"- **Pass**: {pass_count}/{len(results)} ({100*pass_count/len(results):.1f}%)\n")
        f.write(f"- **Partial**: {partial_count}/{len(results)} ({100*partial_count/len(results):.1f}%)\n")
        f.write(f"- **Fail**: {fail_count}/{len(results)} ({100*fail_count/len(results):.1f}%)\n\n")

        # Average improvements
        bg_deltas = [r["background_delta"] for r in results if r["background_delta"] is not None]
        align_deltas = [r["alignment_delta"] for r in results if r["alignment_delta"] is not None]
        contrast_deltas = [r["contrast_delta"] for r in results if r["contrast_delta"] is not None]

        f.write("## Average Improvements\n\n")
        if bg_deltas:
            avg_bg = sum(bg_deltas) / len(bg_deltas)
            f.write(f"- **Background**: {avg_bg:+.2f} points (n={len(bg_deltas)})\n")
        if align_deltas:
            avg_align = sum(align_deltas) / len(align_deltas)
            f.write(f"- **Alignment**: {avg_align:+.2f} points (n={len(align_deltas)})\n")
        if contrast_deltas:
            avg_contrast = sum(contrast_deltas) / len(contrast_deltas)
            f.write(f"- **Contrast**: {avg_contrast:+.2f} points (n={len(contrast_deltas)})\n")

        f.write("\n")

        # Detailed table
        f.write("## Detailed Results\n\n")
        f.write("| Image | Background Δ | Alignment Δ | Contrast Δ | Success |\n")
        f.write("|-------|--------------|-------------|------------|--------|\n")

        for r in results:
            bg = f"{r['background_delta']:+.1f}" if r['background_delta'] is not None else "N/A"
            align = f"{r['alignment_delta']:+.1f}" if r['alignment_delta'] is not None else "N/A"
            contrast = f"{r['contrast_delta']:+.1f}" if r['contrast_delta'] is not None else "N/A"
            status_emoji = {
                "pass": "✅",
                "partial": "⚠️",
                "fail": "❌",
            }.get(r["success"].lower(), "❓")
            f.write(f"| {r['image_id']} | {bg} | {align} | {contrast} | {status_emoji} {r['success']} |\n")

        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        if fail_count > 0:
            f.write(f"- **{fail_count} failures detected**: Review individual reports for root cause analysis\n")
        if avg_bg < 2.0 if bg_deltas else False:
            f.write("- **Low background improvement**: Consider alternative normalization method\n")
        if avg_align < 3.0 if align_deltas else False:
            f.write("- **Low alignment improvement**: Verify deskewing angle detection accuracy\n")
        if pass_count / len(results) < 0.8:
            f.write("- **Success rate <80%**: Phase validation criteria not met, investigate issues\n")
        else:
            f.write("- **Success rate ≥80%**: Phase validation criteria met ✅\n")

    print(f"Summary written to: {output_file}")
    print(f"Pass rate: {pass_count}/{len(results)} ({100*pass_count/len(results):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate VLM validation reports")
    parser.add_argument("--input", type=Path, required=True, help="Directory with validation reports")
    parser.add_argument("--output", type=Path, required=True, help="Output summary file")
    args = parser.parse_args()

    aggregate_validations(args.input, args.output)
