#!/usr/bin/env python3
import argparse
import datetime
import re
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from experiment_tracker.core import ExperimentTracker
from experiment_tracker.templates import TemplateRegistry


def _slugify(text: str) -> str:
    """Convert text to URL-friendly slug for filename."""
    # Convert to lowercase and replace spaces/underscores with hyphens
    slug = text.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special chars
    slug = re.sub(r'[-\s]+', '-', slug)  # Replace spaces/underscores with hyphens
    return slug


def main():
    parser = argparse.ArgumentParser(description="Generate an assessment for the current experiment")
    parser.add_argument(
        "--template",
        required=True,
        help="Template id (registered in .templates/assessment_templates.json) or free-form title",
    )
    parser.add_argument("--title", help="Assessment title (defaults to template name)")
    parser.add_argument("--verbose", default="minimal", help="Verbosity level")
    parser.add_argument("--output", help="Output file path (auto-generated if not provided)")

    args = parser.parse_args()

    tracker = ExperimentTracker()
    experiment_id = tracker._get_current_experiment_id()
    if not experiment_id:
        print("No active experiment found.")
        sys.exit(1)

    registry = TemplateRegistry(tracker.root_dir, tracker.config)
    template_meta = registry.get_assessment_template(args.template)

    title = args.title or (template_meta.title if template_meta else args.template)

    # Get current timestamp in KST
    now = datetime.datetime.now()
    timestamp_str = tracker._format_timestamp(now)

    # Generate filename with format: YYYYMMDD_HHMM-filename.md
    filename_format = tracker.config.get("assessment_filename_format", "%Y%m%d_%H%M")
    timestamp_prefix = now.strftime(filename_format)

    # Create slug from template/title
    slug = _slugify(title)
    filename = f"{timestamp_prefix}-{slug}.md"

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        assessments_dir = tracker.experiments_dir / experiment_id / "assessments"
        assessments_dir.mkdir(parents=True, exist_ok=True)
        output_path = assessments_dir / filename

    # Generate frontmatter and content
    frontmatter_lines = [
        "---",
        f'title: "{title}"',
        f'date: "{timestamp_str}"',
        'status: "draft"',
        f'author: "{tracker.config.get("default_author", "AI Agent")}"',
    ]

    if template_meta:
        frontmatter_lines.append(f'template_id: "{template_meta.id}"')
        # If the template id looks like a run log, tag it as such for discovery.
        if template_meta.id.startswith("run-log"):
            frontmatter_lines.append('kind: "run_log"')

    frontmatter_lines.append("---\n")
    frontmatter = "\n".join(frontmatter_lines)

    header = f"# {title}\n\n"
    header += f"**Date**: {timestamp_str}\n"
    header += "**Status**: Draft\n"
    header += f"**Author**: {tracker.config.get('default_author', 'AI Agent')}\n\n"

    # Add experiment context if verbose
    if args.verbose != "minimal":
        header += f"**Experiment ID**: {experiment_id}\n"
        header += f"**Verbosity**: {args.verbose}\n\n"

    if template_meta:
        template_body = template_meta.path.read_text().strip() + "\n"
    else:
        template_body = "## Findings\n\n(Auto-generated placeholder)\n"

    with open(output_path, "w") as f:
        f.write(frontmatter + header + template_body)

    print(f"Generated assessment at {output_path}")


if __name__ == "__main__":
    main()
