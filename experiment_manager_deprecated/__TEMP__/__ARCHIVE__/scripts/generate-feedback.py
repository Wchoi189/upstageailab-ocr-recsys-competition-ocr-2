#!/usr/bin/env python3
"""Generate experiment feedback log from template."""

import argparse
from datetime import datetime
from pathlib import Path

import yaml


def get_current_experiment():
    """Get current experiment from .current symlink."""
    exp_tracker = Path(__file__).parent.parent
    current_link = exp_tracker / "experiments" / ".current"

    if not current_link.exists():
        return None

    # Read symlink target
    target = current_link.resolve()
    if not target.exists():
        return None

    return target


def load_experiment_state(exp_dir: Path):
    """Load experiment state.yml."""
    state_file = exp_dir / "state.yml"
    if not state_file.exists():
        return None

    with open(state_file) as f:
        return yaml.safe_load(f)


def generate_feedback_log(exp_dir: Path, agent_id: str = "github_copilot"):
    """Generate feedback log from template."""
    template_path = Path(__file__).parent.parent / ".templates" / "feedback_template.md"

    if not template_path.exists():
        print(f"Error: Template not found at {template_path}")
        return None

    # Load template
    with open(template_path) as f:
        template = f.read()

    # Load experiment state
    state = load_experiment_state(exp_dir)
    if not state:
        print(f"Warning: Could not load experiment state from {exp_dir}")
        exp_id = exp_dir.name
        exp_type = "unknown"
    else:
        exp_id = state.get("id", exp_dir.name)
        exp_type = state.get("type", "unknown")

    # Get current timestamp
    now = datetime.now()
    timestamp_kst = now.strftime("%Y-%m-%d %H:%M (KST)")
    date_str = now.strftime("%Y-%m-%d")

    # Replace placeholders
    feedback = template.replace("YYYY-MM-DD HH:MM (KST)", timestamp_kst)
    feedback = feedback.replace("YYYYMMDD_HHMMSS_type", exp_id)
    feedback = feedback.replace("github_copilot", agent_id)
    feedback = feedback.replace("[Auto-filled from metadata]", exp_id)
    feedback = feedback.replace("[e.g., perspective_correction, ocr_training]", exp_type)
    feedback = feedback.replace("[YYYY-MM-DD]", date_str)

    return feedback


def main():
    parser = argparse.ArgumentParser(description="Generate experiment feedback log from template")
    parser.add_argument(
        "--experiment-id",
        help="Experiment ID (default: current experiment from .current symlink)",
    )
    parser.add_argument(
        "--agent-id",
        default="github_copilot",
        help="Agent ID (default: github_copilot)",
    )
    parser.add_argument(
        "--output",
        help="Output file path (default: <experiment_dir>/feedback_log.md)",
    )

    args = parser.parse_args()

    # Determine experiment directory
    if args.experiment_id:
        exp_tracker = Path(__file__).parent.parent
        exp_dir = exp_tracker / "experiments" / args.experiment_id
        if not exp_dir.exists():
            print(f"Error: Experiment directory not found: {exp_dir}")
            return 1
    else:
        exp_dir = get_current_experiment()
        if not exp_dir:
            print("Error: No current experiment found. Use --experiment-id to specify.")
            return 1

    print(f"Generating feedback log for experiment: {exp_dir.name}")

    # Generate feedback log
    feedback = generate_feedback_log(exp_dir, args.agent_id)
    if not feedback:
        return 1

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = exp_dir / "feedback_log.md"

    # Write feedback log
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(feedback)

    print(f"✅ Feedback log created: {output_path}")
    print("\nNext steps:")
    print(f"1. Edit {output_path} with experiment-specific feedback")
    print("2. Submit feedback for review")
    print("3. Track improvements in experiment_manager/WORKFLOW_IMPROVEMENTS_SUMMARY.md")

    return 0


if __name__ == "__main__":
    exit(main())
