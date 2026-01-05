#!/usr/bin/env python3
"""
Sync Project Compass Active Tasks to GitHub Projects.
Usage: python AgentQMS/tools/utils/sync_github_projects.py [--init] [--roadmap <file>] [--dry-run]
"""

import argparse
import sys
import yaml
import subprocess
import shutil
import json
from pathlib import Path

COMPASS_SESSION_PATH = Path("project_compass/active_context/current_session.yml")

class GitHubManager:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.session_data = self._load_session()
        self._check_dependencies()

    def _load_session(self):
        if not COMPASS_SESSION_PATH.exists():
            print(f"Error: Session file not found at {COMPASS_SESSION_PATH}")
            sys.exit(1)
        with open(COMPASS_SESSION_PATH) as f:
            return yaml.safe_load(f)

    def _check_dependencies(self):
        if not shutil.which("gh"):
            print("Error: GitHub CLI (gh) not found in PATH.")
            sys.exit(1)

        # Check auth status (optional but good practice)
        # self.run_command(["gh", "auth", "status"], check=False)

    def run_command(self, cmd, check=True):
        if self.dry_run:
            print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
            return True, ""

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {' '.join(cmd)}")
            print(f"Stderr: {e.stderr}")
            if check:
                raise
            return False, e.stderr

    def find_project_by_title(self, title):
        """Find a GitHub Project V2 by title."""
        print(f"Searching for project: {title}...")
        # Note: This requires 'project' scope.
        # listing projects might default to user unless owner is specified.
        # We'll try to list viewer's projects.
        cmd = ["gh", "project", "list", "--format", "json", "--limit", "50"]

        success, output = self.run_command(cmd, check=False)
        if not success and not self.dry_run:
             # Fallback or specific error handling
             return None

        if self.dry_run:
            return None # Simulate not found for init

        try:
            projects = json.loads(output)
            # projects is a list of dicts: { "title": "...", "number": 1, ... }
            for p in projects.get('projects', []): # gh project list output structure varies by version, usually direct list or dict
                # Adjust based on actual gh cli output for 'project list --format json' which is usually a list under 'projects' key or direct list
                # Actually `gh project list --format json` output is `{"projects": [...]}`
                if p.get('title') == title:
                    return p
            return None
        except json.JSONDecodeError:
            print("Error parsing project list JSON.")
            return None

    def find_issue_by_title(self, title):
        """Check if an issue with the given title exists to avoid duplicates."""
        # Using gh issue list --search
        cmd = ["gh", "issue", "list", "--search",f"{title} in:title", "--json", "title,url", "--limit", "1"]
        success, output = self.run_command(cmd, check=False)

        if self.dry_run:
            return None

        if success and output:
            try:
                issues = json.loads(output)
                if issues:
                    return issues[0]
            except json.JSONDecodeError:
                pass
        return None

    def get_current_user(self):
        """Get the currently authenticated GitHub user."""
        if self.dry_run:
            return "dry-run-user"

        # Try `gh api user --jq .login`
        cmd = ["gh", "api", "user", "--jq", ".login"]
        success, output = self.run_command(cmd, check=False)
        if success and output:
            return output.strip()

        # Fallback?
        print("Error: Could not determine current GitHub user. Please ensure `gh auth status` is healthy.")
        return None

    def create_project(self, title):
        owner = self.get_current_user()
        if not owner:
            return None

        print(f"Creating Project: {title} (Owner: {owner})")
        cmd = ["gh", "project", "create", "--title", title, "--owner", owner, "--format", "json"]
        success, output = self.run_command(cmd)
        if success and not self.dry_run:
            try:
                return json.loads(output)
            except:
                return None
        return None

    def ensure_label(self, name, color="ededed", description=""):
        """Ensure a label exists in the repository."""
        if self.dry_run:
            return

        print(f"Ensuring label exists: {name}")
        # Check if label exists
        cmd = ["gh", "label", "list", "--search", name, "--json", "name"]
        success, output = self.run_command(cmd, check=False)

        exists = False
        if success and output:
            try:
                labels = json.loads(output)
                for l in labels:
                    if l['name'] == name:
                        exists = True
                        break
            except json.JSONDecodeError:
                pass

        if not exists:
            print(f"Creating label: {name}")
            create_cmd = ["gh", "label", "create", name, "--color", color, "--description", description]
            self.run_command(create_cmd, check=False)

    def create_issue(self, title, body, labels=None):
        print(f"Creating Issue: {title}")

        if labels:
            for lbl in labels:
                self.ensure_label(lbl, color="7F39FB", description="Created by Project Compass")

        cmd = ["gh", "issue", "create", "--title", title, "--body", body]
        if labels:
            for lbl in labels:
                cmd.extend(["--label", lbl])

        self.run_command(cmd)

    def sync(self, init=False, roadmap_path=None):
        session_id = self.session_data.get('session_id', 'Session')
        objective = self.session_data.get('objective', 'Objective')
        project_title = f"{session_id} - {objective}"

        project = None
        if init:
            project = self.find_project_by_title(project_title)
            if not project:
                project = self.create_project(project_title)
            else:
                print(f"Project '{project_title}' already exists.")

        if roadmap_path:
            self.publish_roadmap(roadmap_path, session_id)

        # Sync Tasks
        tasks = self.session_data.get('active_tasks', [])
        print(f"Syncing {len(tasks)} tasks...")
        for task in tasks:
            # Check for generic duplicates? Or just create
            # For this iteration, we'll try to find if we already posted it
            issue_title = f"[Compass] {task}"

            existing = self.find_issue_by_title(issue_title)
            if existing:
                print(f"Skipping existing issue: {issue_title}")
                continue

            self.create_issue(
                title=issue_title,
                body=f"Task from Project Compass Session: {session_id}",
                labels=["compass-task"]
            )

    def publish_roadmap(self, roadmap_path, session_id):
        path = Path(roadmap_path)
        if not path.exists():
            print(f"Warning: Roadmap file not found at {path}")
            return

        with open(path) as f:
            content = f.read()

        title = f"Implementation Roadmap: {session_id}"
        existing = self.find_issue_by_title(title)

        if existing:
            print(f"Roadmap issue already exists: {existing.get('url')}")
            # Optional: Update it? For now just skip
            return

        print(f"Publishing Roadmap from {roadmap_path}")
        self.create_issue(
            title=title,
            body=content,
            labels=["roadmap", "compass-plan"]
        )

def main():
    parser = argparse.ArgumentParser(description="Sync Compass tasks to GitHub")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--init", action="store_true", help="Initialize a new GitHub Project for this session")
    parser.add_argument("--roadmap", type=str, help="Path to roadmap/implementation plan artifact to publish as an issue")

    args = parser.parse_args()

    manager = GitHubManager(dry_run=args.dry_run)
    manager.sync(init=args.init, roadmap_path=args.roadmap)

if __name__ == "__main__":
    main()

