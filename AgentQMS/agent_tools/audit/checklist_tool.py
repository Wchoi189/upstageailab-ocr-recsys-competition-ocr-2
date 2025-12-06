#!/usr/bin/env python3
"""
Audit Framework Checklist Tool

Generates, tracks, and reports on audit framework checklists.

Usage:
    python checklist_tool.py generate --phase "discovery" --output "docs/audit/checklist_discovery.md"
    python checklist_tool.py track --checklist "docs/audit/checklist_discovery.md" --item "Task 1.1" --status "completed"
    python checklist_tool.py report --audit-dir "docs/audit"
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List

from AgentQMS.agent_tools.utils.runtime import ensure_project_root_on_sys_path
from AgentQMS.agent_tools.utils.paths import get_project_conventions_dir

ensure_project_root_on_sys_path()


def get_protocol_dir() -> Path:
    """Get the audit framework protocol directory."""
    protocol_dir = get_project_conventions_dir() / "audit_framework" / "protocol"
    if protocol_dir.exists():
        return protocol_dir
    raise RuntimeError(f"Protocol directory not found: {protocol_dir}")


def extract_checklist_for_phase(phase: str) -> Dict[str, List[str]]:
    """
    Extract checklist items for a specific phase from checklists.md.
    
    Args:
        phase: Phase name (discovery, analysis, design, implementation, automation)
    
    Returns:
        Dictionary mapping category names to list of checklist items
    """
    protocol_dir = get_protocol_dir()
    checklists_file = protocol_dir / "checklists.md"
    
    if not checklists_file.exists():
        raise FileNotFoundError(f"Checklists file not found: {checklists_file}")
    
    content = checklists_file.read_text(encoding="utf-8")
    
    # Find the phase section
    phase_pattern = rf"## Phase \d+: {phase.title()} Checklist"
    phase_match = re.search(phase_pattern, content, re.IGNORECASE)
    
    if not phase_match:
        # Try alternative pattern
        phase_pattern = rf"## {phase.title()} Checklist"
        phase_match = re.search(phase_pattern, content, re.IGNORECASE)
    
    if not phase_match:
        available_phases = ["discovery", "analysis", "design", "implementation", "automation"]
        raise ValueError(
            f"Phase '{phase}' not found in checklists.\n"
            f"Available phases: {', '.join(available_phases)}"
        )
    
    # Extract content from phase section to next phase or end
    start_pos = phase_match.end()
    next_phase_match = re.search(r"## Phase \d+:", content[start_pos:])
    if next_phase_match:
        phase_content = content[start_pos:start_pos + next_phase_match.start()]
    else:
        phase_content = content[start_pos:]
    
    # Extract categories and items
    categories = {}
    current_category = None
    
    # Find category headers (### Category Name)
    category_pattern = r"### ([^\n]+)"
    item_pattern = r"- \[ \] (.+)"
    
    for line in phase_content.split("\n"):
        category_match = re.match(category_pattern, line)
        if category_match:
            current_category = category_match.group(1).strip()
            categories[current_category] = []
        elif current_category:
            item_match = re.match(item_pattern, line)
            if item_match:
                item = item_match.group(1).strip()
                categories[current_category].append(item)
    
    return categories


def generate_checklist(phase: str, output_path: Path) -> Path:
    """
    Generate a checklist file for a specific phase.
    
    Args:
        phase: Phase name
        output_path: Path where checklist should be written
    
    Returns:
        Path to generated checklist
    """
    categories = extract_checklist_for_phase(phase)
    
    # Generate checklist content
    lines = [
        f"# {phase.title()} Phase Checklist",
        "",
        f"**Phase**: {phase.title()}",
        f"**Generated**: {Path.cwd()}",
        "",
        "---",
        "",
    ]
    
    for category, items in categories.items():
        lines.append(f"## {category}")
        lines.append("")
        for item in items:
            lines.append(f"- [ ] {item}")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("**Status**: In Progress")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write checklist
    output_path.write_text("\n".join(lines), encoding="utf-8")
    
    print(f"✅ Generated checklist: {output_path}")
    return output_path


def update_checklist_item(checklist_path: Path, item_text: str, status: str) -> None:
    """
    Update a checklist item status.
    
    Args:
        checklist_path: Path to checklist file
        item_text: Text of the checklist item to update
        status: New status ("completed" or "pending")
    """
    if not checklist_path.exists():
        raise FileNotFoundError(f"Checklist not found: {checklist_path}")
    
    content = checklist_path.read_text(encoding="utf-8")
    
    # Find and replace the item
    if status.lower() in ["completed", "done", "x"]:
        marker = "[x]"
    else:
        marker = "[ ]"
    
    # Pattern to match checklist item
    pattern = rf"(- \[[ x]\] ){re.escape(item_text)}"
    replacement = f"- {marker} {item_text}"
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print(f"⚠️  Item not found: {item_text}")
        return
    
    checklist_path.write_text(new_content, encoding="utf-8")
    print(f"✅ Updated: {item_text} -> {status}")


def generate_progress_report(audit_dir: Path) -> str:
    """
    Generate a progress report for all checklists in audit directory.
    
    Args:
        audit_dir: Directory containing audit documents and checklists
    
    Returns:
        Progress report as string
    """
    checklist_files = list(audit_dir.glob("checklist_*.md"))
    
    if not checklist_files:
        return "No checklists found in audit directory."
    
    report_lines = [
        "# Audit Progress Report",
        "",
        f"**Generated**: {Path.cwd()}",
        "",
        "---",
        "",
    ]
    
    total_items = 0
    completed_items = 0
    
    for checklist_path in sorted(checklist_files):
        content = checklist_path.read_text(encoding="utf-8")
        
        # Count items
        all_items = re.findall(r"- \[([ x])\] (.+)", content)
        phase_total = len(all_items)
        phase_completed = sum(1 for marker, _ in all_items if marker == "x")
        
        total_items += phase_total
        completed_items += phase_completed
        
        phase_name = checklist_path.stem.replace("checklist_", "").title()
        percentage = (phase_completed / phase_total * 100) if phase_total > 0 else 0
        
        report_lines.append(f"## {phase_name} Phase")
        report_lines.append(f"- **Progress**: {phase_completed}/{phase_total} ({percentage:.1f}%)")
        report_lines.append(f"- **Checklist**: {checklist_path.name}")
        report_lines.append("")
    
    overall_percentage = (completed_items / total_items * 100) if total_items > 0 else 0
    report_lines.extend([
        "---",
        "",
        "## Overall Progress",
        f"- **Total Items**: {total_items}",
        f"- **Completed**: {completed_items}",
        f"- **Remaining**: {total_items - completed_items}",
        f"- **Progress**: {overall_percentage:.1f}%",
        "",
    ])
    
    return "\n".join(report_lines)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Manage audit framework checklists",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate checklist for phase
  python checklist_tool.py generate --phase "discovery" --output "docs/audit/checklist_discovery.md"
  
  # Update checklist item
  python checklist_tool.py track --checklist "docs/audit/checklist_discovery.md" \\
      --item "Scan for broken dependencies" --status "completed"
  
  # Generate progress report
  python checklist_tool.py report --audit-dir "docs/audit"
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate checklist for phase")
    generate_parser.add_argument(
        "--phase",
        required=True,
        choices=["discovery", "analysis", "design", "implementation", "automation"],
        help="Phase name"
    )
    generate_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file path"
    )
    
    # Track command
    track_parser = subparsers.add_parser("track", help="Update checklist item status")
    track_parser.add_argument(
        "--checklist",
        type=Path,
        required=True,
        help="Path to checklist file"
    )
    track_parser.add_argument(
        "--item",
        required=True,
        help="Text of checklist item to update"
    )
    track_parser.add_argument(
        "--status",
        required=True,
        choices=["completed", "pending"],
        help="New status"
    )
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate progress report")
    report_parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("docs/audit"),
        help="Directory containing audit documents (default: docs/audit)"
    )
    report_parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "generate":
            generate_checklist(args.phase, args.output)
        elif args.command == "track":
            update_checklist_item(args.checklist, args.item, args.status)
        elif args.command == "report":
            report = generate_progress_report(args.audit_dir)
            if args.output:
                args.output.write_text(report, encoding="utf-8")
                print(f"✅ Generated report: {args.output}")
            else:
                print(report)
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

