#!/usr/bin/env python3
"""
Plan Progress Tracker for AgentQMS

Updates Progress Tracker sections in implementation plans and other
markdown documents with markdown checklist syntax.

Supports updating:
- STATUS field
- CURRENT_STEP field
- NEXT_TASK field
- Checklist items (- [ ] / - [x])

Usage:
    python plan_progress.py update-status --file plan.md --status "In Progress"
    python plan_progress.py update-step --file plan.md --step "Phase 2, Task 2.3"
    python plan_progress.py mark-complete --file plan.md --task "Task 2.2"
    python plan_progress.py show --file plan.md
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()


class PlanProgressTracker:
    """Manages progress tracking in implementation plan markdown files."""

    def __init__(self, file_path: Path):
        """Initialize the tracker.

        Args:
            file_path: Path to the markdown file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self.content = self.file_path.read_text(encoding="utf-8")
        self.original_content = self.content

    def update_status(self, status: str) -> bool:
        """Update STATUS field in Progress Tracker section.

        Args:
            status: New status value

        Returns:
            True if updated, False if not found
        """
        pattern = r"(- \*\*STATUS:\*\*\s*)([^\n]+)"
        if re.search(pattern, self.content):
            self.content = re.sub(pattern, rf"\1{status}", self.content)
            return True
        return False

    def update_current_step(self, step: str) -> bool:
        """Update CURRENT_STEP field in Progress Tracker section.

        Args:
            step: New current step value

        Returns:
            True if updated, False if not found
        """
        pattern = r"(- \*\*CURRENT_STEP:\*\*\s*)([^\n]+)"
        if re.search(pattern, self.content):
            self.content = re.sub(pattern, rf"\1{step}", self.content)
            return True
        return False

    def update_last_completed_task(self, task: str) -> bool:
        """Update LAST_COMPLETED_TASK field.

        Args:
            task: Task description

        Returns:
            True if updated, False if not found
        """
        pattern = r"(- \*\*LAST_COMPLETED_TASK:\*\*\s*)([^\n]+)"
        if re.search(pattern, self.content):
            self.content = re.sub(pattern, rf"\1{task}", self.content)
            return True
        return False

    def update_next_task(self, task: str) -> bool:
        """Update NEXT_TASK field.

        Args:
            task: Task description

        Returns:
            True if updated, False if not found
        """
        pattern = r"(- \*\*NEXT_TASK:\*\*\s*)([^\n]+)"
        if re.search(pattern, self.content):
            self.content = re.sub(pattern, rf"\1{task}", self.content)
            return True
        return False

    def mark_checklist_complete(self, task_pattern: str) -> int:
        """Mark checklist items matching a pattern as complete.

        Matches task patterns like "Task 2.1" or "Task 2.1: Plugin Registry"

        Args:
            task_pattern: Pattern to match (e.g., "Task 2.1")

        Returns:
            Number of items updated
        """
        count = 0

        # Pattern 1: Main task header (e.g., "- [x] **Task 2.1: Something**")
        header_pattern = (
            rf"(- )\[ \]([ ]+\*\*{re.escape(task_pattern)}[^\n]*\*\*)"
        )
        if re.search(header_pattern, self.content):
            self.content = re.sub(header_pattern, r"\1[x]\2", self.content)
            count += 1

        # Pattern 2: Sub-items (e.g., "- [x] Do something")
        # Find the section for this task and mark subsequent items as complete
        task_match = re.search(
            rf"(- )\[ \]([ ]+{re.escape(task_pattern)}[^\n]*)",
            self.content,
        )
        if task_match:
            start_pos = task_match.start()

            # Find all sub-items after this task until the next main task
            sub_pattern = r"(   - )\[ \]([^\n]*)"
            remaining_content = self.content[start_pos:]

            # Stop at next main task (pattern: "- [ ] **Task X.Y")
            next_task_match = re.search(r"^(- )\[ \](\s+\*\*Task \d+\.\d+)", remaining_content[10:], re.MULTILINE)
            end_pos = next_task_match.start() + 10 if next_task_match else len(remaining_content)

            section = remaining_content[:end_pos]
            updated_section = re.sub(sub_pattern, r"\1[x]\2", section)
            count += len(re.findall(sub_pattern, section))

            # Replace the section in the full content
            self.content = (
                self.content[:start_pos]
                + updated_section
                + remaining_content[end_pos:]
            )

        return count

    def mark_checklist_incomplete(self, task_pattern: str) -> int:
        """Mark checklist items matching a pattern as incomplete.

        Args:
            task_pattern: Pattern to match (e.g., "Task 2.1")

        Returns:
            Number of items updated
        """
        count = 0

        # Pattern 1: Main task header
        header_pattern = (
            rf"(- )\[x\]([ ]+\*\*{re.escape(task_pattern)}[^\n]*\*\*)"
        )
        if re.search(header_pattern, self.content):
            self.content = re.sub(header_pattern, r"\1[ ]\2", self.content)
            count += 1

        # Pattern 2: Sub-items
        sub_pattern = rf"(   - )\[x\]([^\n]*{re.escape(task_pattern)}[^\n]*)"
        matches = len(re.findall(sub_pattern, self.content))
        self.content = re.sub(sub_pattern, r"\1[ ]\2", self.content)
        count += matches

        return count

    def get_status(self) -> dict[str, Any]:
        """Extract progress tracker fields from document.

        Returns:
            Dictionary with extracted values
        """
        status_match = re.search(r"- \*\*STATUS:\*\*\s*(.+)", self.content)
        step_match = re.search(r"- \*\*CURRENT_STEP:\*\*\s*(.+)", self.content)
        completed_match = re.search(
            r"- \*\*LAST_COMPLETED_TASK:\*\*\s*(.+)",
            self.content,
        )
        next_match = re.search(r"- \*\*NEXT_TASK:\*\*\s*(.+)", self.content)

        return {
            "status": status_match.group(1) if status_match else "Unknown",
            "current_step": step_match.group(1) if step_match else "Unknown",
            "last_completed_task": completed_match.group(1)
                if completed_match
                else "Unknown",
            "next_task": next_match.group(1) if next_match else "Unknown",
        }

    def count_completed_tasks(self) -> dict[str, int]:
        """Count completed and total checklist items.

        Returns:
            Dictionary with counts for each phase
        """
        # Find all main task headers
        completed = len(re.findall(r"- \[x\]\s+\*\*Task \d+\.\d+:", self.content))
        total = len(re.findall(r"- \[ \]\s+\*\*Task \d+\.\d+:|"
                              r"- \[x\]\s+\*\*Task \d+\.\d+:", self.content))

        return {
            "completed_tasks": completed,
            "total_tasks": total,
            "completion_percentage": round(
                (completed / total * 100) if total > 0 else 0
            ),
        }

    def save(self, dry_run: bool = False) -> bool:
        """Save changes to file.

        Args:
            dry_run: If True, don't write to disk

        Returns:
            True if content changed
        """
        if self.content == self.original_content:
            return False

        if not dry_run:
            self.file_path.write_text(self.content, encoding="utf-8")

        return True

    def format_output(self) -> str:
        """Format current progress for display.

        Returns:
            Formatted progress report
        """
        tracker = self.get_status()
        counts = self.count_completed_tasks()

        lines = []
        lines.append("📊 Plan Progress Summary")
        lines.append("=" * 50)
        lines.append(f"Status: {tracker['status']}")
        lines.append(f"Current Step: {tracker['current_step']}")
        lines.append(f"Last Completed: {tracker['last_completed_task']}")
        lines.append(f"Next Task: {tracker['next_task']}")
        lines.append("")
        lines.append(f"Tasks Completed: {counts['completed_tasks']}/{counts['total_tasks']} "
                    f"({counts['completion_percentage']}%)")

        return "\n".join(lines)


def main() -> int:
    """Command-line interface for plan progress tracking."""
    parser = argparse.ArgumentParser(
        description="Track and update implementation plan progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current progress
  %(prog)s show --file plan.md

  # Update status
  %(prog)s update-status --file plan.md --status "Phase 2 In Progress"

  # Mark a task as complete
  %(prog)s mark-complete --file plan.md --task "Task 2.1"

  # Update current step
  %(prog)s update-step --file plan.md --step "Phase 2, Task 2.3"

  # Dry run (see changes without writing)
  %(prog)s mark-complete --file plan.md --task "Task 2.1" --dry-run
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show progress tracker")
    show_parser.add_argument("--file", "-f", type=Path, required=True,
                           help="Path to plan file")

    # Update status
    status_parser = subparsers.add_parser("update-status",
                                         help="Update STATUS field")
    status_parser.add_argument("--file", "-f", type=Path, required=True)
    status_parser.add_argument("--status", "-s", required=True,
                             help="New status value")
    status_parser.add_argument("--dry-run", action="store_true",
                             help="Show changes without writing")

    # Update current step
    step_parser = subparsers.add_parser("update-step",
                                       help="Update CURRENT_STEP field")
    step_parser.add_argument("--file", "-f", type=Path, required=True)
    step_parser.add_argument("--step", "-s", required=True,
                           help="New step value")
    step_parser.add_argument("--dry-run", action="store_true")

    # Mark complete
    complete_parser = subparsers.add_parser("mark-complete",
                                           help="Mark task as complete")
    complete_parser.add_argument("--file", "-f", type=Path, required=True)
    complete_parser.add_argument("--task", "-t", required=True,
                               help="Task pattern (e.g., 'Task 2.1')")
    complete_parser.add_argument("--dry-run", action="store_true")

    # Mark incomplete
    incomplete_parser = subparsers.add_parser("mark-incomplete",
                                             help="Mark task as incomplete")
    incomplete_parser.add_argument("--file", "-f", type=Path, required=True)
    incomplete_parser.add_argument("--task", "-t", required=True)
    incomplete_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        tracker = PlanProgressTracker(args.file)

        if args.command == "show":
            print(tracker.format_output())

        elif args.command == "update-status":
            if tracker.update_status(args.status):
                if tracker.save(dry_run=args.dry_run):
                    if args.dry_run:
                        print("✅ Would update STATUS (dry-run mode)")
                    else:
                        print(f"✅ Updated STATUS to: {args.status}")
                else:
                    print("⚠️  No changes made")
            else:
                print("❌ STATUS field not found in document")
                return 1

        elif args.command == "update-step":
            if tracker.update_current_step(args.step):
                if tracker.save(dry_run=args.dry_run):
                    if args.dry_run:
                        print("✅ Would update CURRENT_STEP (dry-run mode)")
                    else:
                        print(f"✅ Updated CURRENT_STEP to: {args.step}")
                else:
                    print("⚠️  No changes made")
            else:
                print("❌ CURRENT_STEP field not found in document")
                return 1

        elif args.command == "mark-complete":
            count = tracker.mark_checklist_complete(args.task)
            if count > 0:
                if tracker.save(dry_run=args.dry_run):
                    if args.dry_run:
                        print(f"✅ Would mark {count} items as complete (dry-run)")
                    else:
                        print(f"✅ Marked {count} checklist items as complete")
                else:
                    print("⚠️  No changes made")
            else:
                print(f"❌ No checklist items found matching '{args.task}'")
                return 1

        elif args.command == "mark-incomplete":
            count = tracker.mark_checklist_incomplete(args.task)
            if count > 0:
                if tracker.save(dry_run=args.dry_run):
                    if args.dry_run:
                        print(f"✅ Would mark {count} items as incomplete (dry-run)")
                    else:
                        print(f"✅ Marked {count} checklist items as incomplete")
                else:
                    print("⚠️  No changes made")
            else:
                print(f"❌ No checklist items found matching '{args.task}'")
                return 1

        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
