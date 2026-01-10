#!/usr/bin/env python3
"""
Debugging Session Initializer
Creates standardized folder structure for bug investigation sessions.

Usage:
    python init_debug_session.py --id 001 --title "overlay-misalignment"
    python init_debug_session.py --id 002 --title "preprocessing-failure" --severity high

Makefile:
    make debug-session BUG_ID=001 TITLE="description"
"""

import argparse
from datetime import datetime
from pathlib import Path

BASE_PATH = Path("docs/artifacts/bug_reports")

INTAKE_TEMPLATE = """---
schema: debugging-session-v1
schema_version: "1.0"
session_id: {session_id}
phase: 1
status: active
created: "{timestamp}"
severity: {severity}
tags: []
---

# {title}

## Problem Statement
<!-- What is the observable issue? -->

## Symptoms
<!-- What symptoms indicate the bug? -->

## Reproduction Steps
1.
2.
3.

## Environment
- **Branch**:
- **Commit**:
- **Date**: {date}

## Initial Observations
<!-- First impressions, quick investigation results -->
"""

INVESTIGATION_TEMPLATE = """---
schema: debugging-session-v1
schema_version: "1.0"
session_id: {session_id}
phase: 2
status: active
created: "{timestamp}"
severity: {severity}
tags: []
---

# Investigation: {title}

## Hypotheses

### Hypothesis 1
**Statement**:
**Test**:
**Result**:

---

## Key Findings
<!-- Document significant discoveries during investigation -->

## Code References
<!-- Links to relevant code sections -->

## Next Steps
- [ ]
"""

README_TEMPLATE = """# {session_id}: {title}

**Status**: 🟡 Active
**Severity**: {severity}
**Created**: {date}

## Documents

| Phase | Document | Status |
|-------|----------|--------|
| 1 | [01_intake.md](01_intake.md) | ✅ Created |
| 2 | [02_investigation.md](02_investigation.md) | 🟡 In Progress |
| 3 | 03_resolution.md | ⬜ Pending |
| 4 | 04_postmortem.md | ⬜ Pending |

## Artifacts
- [images/](artifacts/images/) - Screenshots, diagrams
- [logs/](artifacts/logs/) - Log excerpts
- [scripts/](artifacts/scripts/) - Reproduction/fix scripts

## Quick Reference
```bash
# View this session
cat docs/artifacts/bug_reports/{session_id}/README.md

# Add screenshot
cp /path/to/screenshot.png docs/artifacts/bug_reports/{session_id}/artifacts/images/

# Add log excerpt
cp /path/to/debug.log docs/artifacts/bug_reports/{session_id}/artifacts/logs/
```
"""


def create_session(bug_id: str, title: str, severity: str = "medium") -> Path:
    """Create debugging session folder with initial structure."""
    session_id = f"BUG-{bug_id.zfill(3)}"
    session_path = BASE_PATH / session_id

    if session_path.exists():
        print(f"❌ Session {session_id} already exists at {session_path}")
        return session_path

    # Create directories
    session_path.mkdir(parents=True, exist_ok=True)
    (session_path / "artifacts" / "images").mkdir(parents=True)
    (session_path / "artifacts" / "logs").mkdir(parents=True)
    (session_path / "artifacts" / "scripts").mkdir(parents=True)

    # Template variables
    now = datetime.now()
    timestamp = now.isoformat()
    date = now.strftime("%Y-%m-%d")
    title_formatted = title.replace("-", " ").title()

    vars = {
        "session_id": session_id,
        "title": title_formatted,
        "timestamp": timestamp,
        "date": date,
        "severity": severity,
    }

    # Write files
    (session_path / "01_intake.md").write_text(INTAKE_TEMPLATE.format(**vars))
    (session_path / "02_investigation.md").write_text(INVESTIGATION_TEMPLATE.format(**vars))
    (session_path / "README.md").write_text(README_TEMPLATE.format(**vars))

    # Add .gitkeep to empty dirs
    (session_path / "artifacts" / "images" / ".gitkeep").touch()
    (session_path / "artifacts" / "logs" / ".gitkeep").touch()
    (session_path / "artifacts" / "scripts" / ".gitkeep").touch()

    print(f"✅ Created debugging session: {session_path}")
    print("   📄 01_intake.md - Fill in problem statement")
    print("   📄 02_investigation.md - Document hypotheses and tests")
    print("   📁 artifacts/ - Store screenshots, logs, scripts")

    return session_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize debugging session")
    parser.add_argument("--id", required=True, help="Bug ID (e.g., 001)")
    parser.add_argument("--title", required=True, help="Short title (e.g., overlay-misalignment)")
    parser.add_argument(
        "--severity", default="medium", choices=["critical", "high", "medium", "low"], help="Bug severity (default: medium)"
    )

    args = parser.parse_args()
    create_session(args.id, args.title, args.severity)


if __name__ == "__main__":
    main()
