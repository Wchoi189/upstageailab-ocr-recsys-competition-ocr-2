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
    slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special chars
    slug = re.sub(r'[-\s]+', '-', slug)  # Replace spaces/underscores with hyphens
    return slug


def main():
    parser = argparse.ArgumentParser(description="Generate an assessment for the current experiment")
    parser.add_argument("--template", required=True, help="Template name or assessment title")
    parser.add_argument("--title", help="Assessment title (defaults to template name)")
    parser.add_argument("--verbose", default="minimal", help="Verbosity level")
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

    # Create slug from template/title
    title = args.title or args.template
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
    frontmatter = f"""---
title: "{title}"
date: "{timestamp_str}"
status: "draft"
author: "{tracker.config.get('default_author', 'AI Agent')}"
---

# {title}

**Date**: {timestamp_str}
**Status**: Draft
**Author**: {tracker.config.get('default_author', 'AI Agent')}

## Findings

(Auto-generated placeholder)

"""

    # Add experiment context if verbose
    if args.verbose != "minimal":
        frontmatter += f"**Experiment ID**: {experiment_id}\n"
        frontmatter += f"**Verbosity**: {args.verbose}\n\n"

    with open(output_path, "w") as f:
        f.write(frontmatter)

    print(f"Generated assessment at {output_path}")


if __name__ == "__main__":
    main()
