import re

def generate_report(results: list[dict]) -> str:
    """Generate a validation report with violation summary table."""
    total_files = len(results)
    valid_files = sum(1 for r in results if r["valid"])
    invalid_files = total_files - valid_files

    report = []
    report.append("=" * 60)
    report.append("ARTIFACT VALIDATION REPORT")
    report.append("=" * 60)
    report.append(f"Total files: {total_files}")
    report.append(f"Valid files: {valid_files}")
    report.append(f"Invalid files: {invalid_files}")
    report.append(f"Compliance rate: {(valid_files / total_files * 100):.1f}%" if total_files > 0 else "N/A")
    report.append("")

    # Generate violation summary table
    if invalid_files > 0:
        # Count violations by type
        violation_counts: dict[str, int] = {}
        for result in results:
            if not result["valid"]:
                for error in result.get("errors", []):
                    # Extract error code if present
                    code_match = re.search(r"\[E(\d+)\]", error)
                    if code_match:
                        code = f"E{code_match.group(1)}"
                    elif "Naming:" in error:
                        code = "Naming"
                    elif "Directory:" in error:
                        code = "Directory"
                    elif "Frontmatter:" in error:
                        code = "Frontmatter"
                    elif "TypeConsistency:" in error:
                        code = "TypeConsistency"
                    elif "Location:" in error:
                        code = "Location"
                    else:
                        code = "Other"

                    violation_counts[code] = violation_counts.get(code, 0) + 1

        report.append("VIOLATION SUMMARY:")
        report.append("-" * 40)
        report.append(f"{'Rule':<20} | {'Count':>6}")
        report.append("-" * 40)
        for code, count in sorted(violation_counts.items(), key=lambda x: -x[1]):
            report.append(f"{code:<20} | {count:>6}")
        report.append("-" * 40)
        report.append("")

        report.append("VIOLATIONS FOUND:")
        report.append("-" * 40)
        for result in results:
            if not result["valid"]:
                report.append(f"\n❌ {result['file']}")
                for error in result["errors"]:
                    report.append(f"   • {error}")

        # Add suggested next command
        report.append("")
        report.append("SUGGESTED NEXT COMMAND:")
        report.append("-" * 40)
        report.append(f'  cd AgentQMS/bin && make fix ARGS="--limit {min(invalid_files, 10)} --dry-run"')
        report.append("")

    if valid_files > 0:
        report.append("\n✅ VALID FILES:")
        report.append("-" * 40)
        for result in results:
            if result["valid"]:
                report.append(f"✓ {result['file']}")

    return "\n".join(report)


def fix_suggestions(results: list[dict]) -> str:
    """Generate fix suggestions for invalid files."""
    suggestions = []
    suggestions.append("FIX SUGGESTIONS:")
    suggestions.append("=" * 40)

    for result in results:
        if not result["valid"]:
            suggestions.append(f"\n📁 {result['file']}")
            for error in result["errors"]:
                if "Naming:" in error:
                    suggestions.append("   🔧 Rename file to follow convention:")
                    suggestions.append("      Format: YYYY-MM-DD_HHMM_{ARTIFACT_TYPE}_descriptive-name.md")
                elif "Directory:" in error:
                    suggestions.append("   🔧 Move file to correct directory")
                elif "Frontmatter:" in error:
                    suggestions.append("   🔧 Add or fix frontmatter:")
                    suggestions.append("      Required fields: title, date, type, category, status, version")

    return "\n".join(suggestions)
