# Spec B — CLI Entry Point (Global Interface)

## Goal
Provide a global CLI entry point so `AgentQMS` can be invoked like a stateless tool (`git`, `uv`) from any working directory.

## In scope
- CLI module (e.g., `AgentQMS/cli.py` or existing CLI updated)
- `pyproject.toml` entry point registration
- Command namespace alignment via canonical module entrypoint: `python -m AgentQMS.cli`

## Acceptance criteria
- Entry point is installed/registered via `pyproject.toml`.
- Running the CLI from an arbitrary directory resolves the correct `project_root` per Spec A.
- At minimum, commands exist for:
  - `status` (or equivalent no-op that prints resolved root)
  - `server start` (or equivalent)
  - `init` delegated to Spec C implementation

## Verification (evidence required)
- Provide command output showing:
  - CLI is invokable from outside the repo
  - Resolved root matches env override + marker traversal cases

## Out of scope
- Root resolution algorithm details (Spec A)
- Scaffolding logic (Spec C)

