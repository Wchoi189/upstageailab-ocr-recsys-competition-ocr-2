#!/usr/bin/env python3
import argparse
import datetime
import re
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_tracker.core import ExperimentTracker


def _slugify(text: str) -> str:
    """Convert text to URL-friendly slug for filename."""
    # Convert to lowercase and replace spaces/underscores with hyphens
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)  # Remove special chars
    slug = re.sub(r"[-\s]+", "-", slug)  # Replace spaces/underscores with hyphens
    return slug


def main():
    parser = argparse.ArgumentParser(description="Generate an incident report for the current experiment")
    parser.add_argument("--title", required=True, help="Incident report title")
    parser.add_argument(
        "--severity", default="medium", choices=["low", "medium", "high", "critical"], help="Severity level (default: medium)"
    )
    parser.add_argument("--status", default="open", choices=["open", "investigating", "resolved", "closed"], help="Status (default: open)")
    parser.add_argument("--tags", help="Comma-separated tags (e.g., 'perspective,corner-detection')")
    parser.add_argument("--related-artifacts", help="Comma-separated artifact paths (e.g., 'artifacts/img1.jpg,artifacts/img2.jpg')")
    parser.add_argument("--related-assessments", help="Comma-separated assessment paths (e.g., 'assessments/20251123_1430-analysis.md')")
    parser.add_argument("--output", help="Output file path (auto-generated if not provided)")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    experiment_id = tracker._get_current_experiment_id()
    if not experiment_id:
        print("No active experiment found.")
        sys.exit(1)

    # Get current timestamp in KST
    now = datetime.datetime.now()
    timestamp_str = tracker._format_timestamp(now)

    # Generate filename with format: YYYYMMDD_HHMM-filename.md
    filename_format = tracker.config.get("assessment_filename_format", "%Y%m%d_%H%M")
    timestamp_prefix = now.strftime(filename_format)

    # Create slug from title
    slug = _slugify(args.title)
    filename = f"{timestamp_prefix}-{slug}.md"

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        paths = tracker._get_paths(experiment_id)
        incident_reports_dir = paths.get_incident_reports_path()
        incident_reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = incident_reports_dir / filename

    # Parse tags
    tags = []
    if args.tags:
        tags = [tag.strip() for tag in args.tags.split(",") if tag.strip()]

    # Parse related artifacts
    related_artifacts = []
    if args.related_artifacts:
        related_artifacts = [path.strip() for path in args.related_artifacts.split(",") if path.strip()]

    # Parse related assessments
    related_assessments = []
    if args.related_assessments:
        related_assessments = [path.strip() for path in args.related_assessments.split(",") if path.strip()]

    # Load template
    template_path = tracker.root_dir / ".templates" / "incident_report.md"
    if template_path.exists():
        with open(template_path) as f:
            template_content = f.read()
    else:
        # Fallback template if file doesn't exist
        template_content = """---
title: "[Short Title, e.g., Perspective Overshoot]"
date: "YYYY-MM-DD HH:MM (KST)"
experiment_id: "YYYYMMDD_HHMM_experiment_type"
severity: "medium"
status: "open"
tags: []
author: "AI Agent"
---

## Defect Analysis: [Short Title, e.g., Perspective Overshoot]

### 1. Visual Artifacts (What does the output look like?)

* **Distortion Type:** [e.g., Shearing, Pincushion, Stretching, Blank Output]

* **Key Features:** [e.g., Text is diagonal, pixel smearing, ROI cropped out]

* **Comparison:** [e.g., Worse than baseline, or regression from previous version]

### 2. Input Characteristics (What is unique about the source?)

* **ROI Coverage:** [e.g., Subject fills 90% of frame]

* **Contrast/Lighting:** [e.g., Low contrast between paper and table]

* **Geometry:** [e.g., Image is already cropped/rectified]

### 3. Geometric/Data Analysis (The Math)

* **Mask Topology:** [e.g., Mask touches all 4 image borders]

* **Corner Detection:** [e.g., Detected points are collinear]

* **Transform Matrix:** [e.g., Matrix appears singular or ill-conditioned]

### 4. Hypothesis & Action Items

* **Theory:** [Why did the logic fail?]

* **Proposed Fix:** [e.g., Add threshold, adjust epsilon, clamp coordinates]

---

## Related Resources

### Related Artifacts

* (No related artifacts)

### Related Assessments

* (No related assessments)
"""

    # Format YAML arrays for frontmatter
    def format_yaml_array(arr):
        if not arr:
            return "[]"
        return "[" + ", ".join(f'"{item}"' for item in arr) + "]"

    # Replace frontmatter placeholders
    frontmatter = f"""---
title: "{args.title}"
date: "{timestamp_str}"
experiment_id: "{experiment_id}"
severity: "{args.severity}"
status: "{args.status}"
tags: {format_yaml_array(tags)}
author: "{tracker.config.get("default_author", "AI Agent")}"
---

"""

    # Replace the title in the template content
    content = template_content.replace(
        "## Defect Analysis: [Short Title, e.g., Perspective Overshoot]", f"## Defect Analysis: {args.title}"
    )

    # Replace frontmatter in template with actual frontmatter
    # Find the frontmatter section (between --- and ---)
    frontmatter_pattern = r"^---\n.*?\n---\n"
    content = re.sub(frontmatter_pattern, frontmatter, content, flags=re.MULTILINE | re.DOTALL)

    # If no frontmatter was found, prepend it
    if not content.startswith("---"):
        content = frontmatter + content

    # Add related artifacts and assessments at the bottom
    # Find the "Related Resources" section and replace the placeholder content
    related_section = "\n---\n\n## Related Resources\n\n### Related Artifacts\n\n"
    if related_artifacts:
        related_section += "\n".join(f"* {artifact}" for artifact in related_artifacts) + "\n"
    else:
        related_section += "* (No related artifacts)\n"

    related_section += "\n### Related Assessments\n\n"
    if related_assessments:
        related_section += "\n".join(f"* {assessment}" for assessment in related_assessments) + "\n"
    else:
        related_section += "* (No related assessments)\n"

    # Replace the Related Resources section
    # Pattern matches from "---" separator through the end of Related Assessments
    related_pattern = r"\n---\n\n## Related Resources\n\n### Related Artifacts\n\n.*?\n### Related Assessments\n\n.*?(\n|$)"
    if re.search(related_pattern, content, flags=re.DOTALL):
        content = re.sub(related_pattern, related_section, content, flags=re.DOTALL)
    else:
        # If section doesn't exist, append it
        content = content.rstrip() + related_section

    # Write the file
    with open(output_path, "w") as f:
        f.write(content)

    # Record in state.json
    relative_path = f"incident_reports/{filename}"
    tracker.record_incident_report(relative_path, experiment_id)

    print(f"Generated incident report at {output_path}")


if __name__ == "__main__":
    main()
