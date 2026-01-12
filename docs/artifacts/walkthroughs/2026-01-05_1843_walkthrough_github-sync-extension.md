# Walkthrough - GitHub Sync Extension

I have successfully implemented the GitHub Sync Extension planning, refactoring the [sync_github_projects.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/sync_github_projects.py) script to be more robust and feature-rich.

## Changes

### [AgentQMS/tools/utils/sync_github_projects.py](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/sync_github_projects.py)

- **Refactoring**: Converted the procedural script into a [GitHubManager](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/sync_github_projects.py#17-179) class.
- **New Features**:
    - `--init`: Checks for an existing GitHub Project V2 by title (Session ID - Objective) and creates one if it doesn't exist.
    - `--roadmap <file>`: Reads a markdown artifact and publishes it as an Issue in the project, with labels "roadmap" and "compass-plan".
- **Enhancements**:
    - **Existence Checks**: Added [find_project_by_title](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/sync_github_projects.py#53-81) and [find_issue_by_title](file:///workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS/tools/utils/sync_github_projects.py#82-99) to prevent duplicates.
    - **Error Handling**: Wrapped subprocess calls in try/except blocks.

## Verification Results

### Dry Run

I performed a dry run using the following command:

```bash
python AgentQMS/tools/utils/sync_github_projects.py --init --roadmap docs/artifacts/implementation_plans/2026-01-05_github_sync_extension_plan.md --dry-run
```

**Output:**

```
Searching for project: 2026-01-05_strategic_refactor - Master Architecture Refactoring (Phase 1 & 2 Planning)...
[DRY RUN] Would execute: gh project list --format json --limit 50
Creating Project: 2026-01-05_strategic_refactor - Master Architecture Refactoring (Phase 1 & 2 Planning)
[DRY RUN] Would execute: gh project create --title 2026-01-05_strategic_refactor - Master Architecture Refactoring (Phase 1 & 2 Planning) --format json
[DRY RUN] Would execute: gh issue list --search Implementation Roadmap: 2026-01-05_strategic_refactor in:title --json title,url --limit 1
Publishing Roadmap from docs/artifacts/implementation_plans/2026-01-05_github_sync_extension_plan.md
Creating Issue: Implementation Roadmap: 2026-01-05_strategic_refactor
[DRY RUN] Would execute: gh issue create --title Implementation Roadmap: 2026-01-05_strategic_refactor --body ... --label roadmap --label compass-plan
Syncing 3 tasks...
...
```

The dry run confirmed that the script:
1.  Correctly constructs the project title.
2.  Checks for project existence.
3.  Proposes creation of the project.
4.  Checks for roadmap issue existence.
5.  Proposes creation of the roadmap issue.
6.  Syncs individual tasks.
