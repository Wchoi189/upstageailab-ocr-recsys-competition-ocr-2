#!/usr/bin/env python3
"""
Agent Feedback Collection System
Collects and manages feedback from AI agents about documentation and system issues
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


class AgentFeedbackCollector:
    """Collects and manages agent feedback about documentation issues."""

    def __init__(self, feedback_dir: str = "docs/artifacts/agent_feedback"):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_file = self.feedback_dir / "feedback_log.json"
        self.suggestions_file = self.feedback_dir / "suggestions.json"

    def collect_feedback(
        self,
        issue_type: str,
        description: str,
        file_path: str | None = None,
        severity: str = "medium",
        suggested_fix: str | None = None,
        agent_context: str | None = None,
    ) -> str:
        """Collect feedback from an agent about a documentation issue."""

        feedback = {
            "timestamp": datetime.now().isoformat(),
            "issue_type": issue_type,
            "description": description,
            "file_path": file_path,
            "severity": severity,
            "suggested_fix": suggested_fix,
            "agent_context": agent_context,
            "status": "open",
            "id": f"FB_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }

        # Load existing feedback
        existing_feedback = self._load_feedback()
        existing_feedback.append(feedback)

        # Save updated feedback
        self._save_feedback(existing_feedback)

        return feedback["id"]

    def suggest_improvement(
        self,
        area: str,
        current_state: str,
        suggested_change: str,
        rationale: str,
        priority: str = "medium",
    ) -> str:
        """Collect improvement suggestions from agents."""

        suggestion = {
            "timestamp": datetime.now().isoformat(),
            "area": area,
            "current_state": current_state,
            "suggested_change": suggested_change,
            "rationale": rationale,
            "priority": priority,
            "status": "pending",
            "id": f"SUG_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        }

        # Load existing suggestions
        existing_suggestions = self._load_suggestions()
        existing_suggestions.append(suggestion)

        # Save updated suggestions
        self._save_suggestions(existing_suggestions)

        return suggestion["id"]

    def report_documentation_issue(
        self,
        file_path: str,
        issue: str,
        impact: str,
        suggested_fix: str | None = None,
    ) -> str:
        """Report a specific documentation issue."""

        return self.collect_feedback(
            issue_type="documentation_issue",
            description=f"Documentation issue in {file_path}: {issue}",
            file_path=file_path,
            severity="high" if "critical" in issue.lower() else "medium",
            suggested_fix=suggested_fix,
            agent_context=f"Impact: {impact}",
        )

    def report_tool_issue(
        self, tool_path: str, error_message: str, context: str
    ) -> str:
        """Report an issue with automation tools."""

        return self.collect_feedback(
            issue_type="tool_issue",
            description=f"Tool issue with {tool_path}: {error_message}",
            file_path=tool_path,
            severity="high",
            suggested_fix="Check tool path and dependencies",
            agent_context=f"Context: {context}",
        )

    def suggest_documentation_improvement(
        self, file_path: str, current_issue: str, improvement: str, rationale: str
    ) -> str:
        """Suggest improvements to documentation."""

        return self.suggest_improvement(
            area=f"Documentation: {file_path}",
            current_state=current_issue,
            suggested_change=improvement,
            rationale=rationale,
            priority="medium",
        )

    def generate_feedback_report(self) -> str:
        """Generate a summary report of all feedback."""
        feedback = self._load_feedback()
        suggestions = self._load_suggestions()

        report = []
        report.append("# Agent Feedback Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Open issues
        open_issues = [f for f in feedback if f["status"] == "open"]
        report.append(f"## Open Issues ({len(open_issues)})")
        for issue in open_issues[-10:]:  # Last 10 issues
            report.append(f"### {issue['id']} - {issue['issue_type']}")
            report.append(f"**Severity**: {issue['severity']}")
            report.append(f"**Description**: {issue['description']}")
            if issue["file_path"]:
                report.append(f"**File**: {issue['file_path']}")
            if issue["suggested_fix"]:
                report.append(f"**Suggested Fix**: {issue['suggested_fix']}")
            report.append("")

        # Pending suggestions
        pending_suggestions = [s for s in suggestions if s["status"] == "pending"]
        report.append(f"## Pending Suggestions ({len(pending_suggestions)})")
        for suggestion in pending_suggestions[-5:]:  # Last 5 suggestions
            report.append(f"### {suggestion['id']} - {suggestion['area']}")
            report.append(f"**Priority**: {suggestion['priority']}")
            report.append(f"**Current State**: {suggestion['current_state']}")
            report.append(f"**Suggested Change**: {suggestion['suggested_change']}")
            report.append(f"**Rationale**: {suggestion['rationale']}")
            report.append("")

        return "\n".join(report)

    def _load_feedback(self) -> list[dict]:
        """Load existing feedback from file."""
        if self.feedback_file.exists():
            with open(self.feedback_file) as f:
                return json.load(f)
        return []

    def _save_feedback(self, feedback: list[dict]):
        """Save feedback to file."""
        with open(self.feedback_file, "w") as f:
            json.dump(feedback, f, indent=2)

    def _load_suggestions(self) -> list[dict]:
        """Load existing suggestions from file."""
        if self.suggestions_file.exists():
            with open(self.suggestions_file) as f:
                return json.load(f)
        return []

    def _save_suggestions(self, suggestions: list[dict]):
        """Save suggestions to file."""
        with open(self.suggestions_file, "w") as f:
            json.dump(suggestions, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Agent Feedback Collection System")
    parser.add_argument(
        "--report", action="store_true", help="Generate feedback report"
    )
    parser.add_argument("--issue", help="Report a documentation issue")
    parser.add_argument("--file", help="File path for the issue")
    parser.add_argument("--description", help="Description of the issue")
    parser.add_argument("--suggest", help="Suggest an improvement")
    parser.add_argument("--area", help="Area for improvement suggestion")
    parser.add_argument("--current", help="Current state for suggestion")
    parser.add_argument("--change", help="Suggested change")
    parser.add_argument("--rationale", help="Rationale for suggestion")

    args = parser.parse_args()

    collector = AgentFeedbackCollector()

    if args.report:
        print(collector.generate_feedback_report())
    elif args.issue and args.description:
        feedback_id = collector.report_documentation_issue(
            file_path=args.file or "unknown",
            issue=args.description,
            impact="Agent workflow affected",
        )
        print(f"Feedback collected with ID: {feedback_id}")
    elif args.suggest and args.area and args.current and args.change and args.rationale:
        suggestion_id = collector.suggest_improvement(
            area=args.area,
            current_state=args.current,
            suggested_change=args.change,
            rationale=args.rationale,
        )
        print(f"Suggestion collected with ID: {suggestion_id}")
    else:
        print("Use --help for usage information")


if __name__ == "__main__":
    main()
