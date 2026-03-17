# Spec C — `init` Scaffolding

## Goal
Enable initializing a project in an empty directory by scaffolding the minimal AgentQMS project structure.

## In scope
- CLI `init` command (or equivalent)
- Create `.agentqms/` in the resolved `project_root` (Spec A fallback-to-CWD supports empty dirs)
- Create minimal required files:
  - `.agentqms/settings.yaml`
  - `.agentqms/registry.yaml`
  - `AGENTS.yaml`
- Optional starter directories only if explicitly enabled by flag/config (avoid clutter by default).

## Acceptance criteria
- `init` succeeds in an empty directory.
- Idempotent by default:
  - If `.agentqms/` exists, do not overwrite without explicit flag.
- All files created in the resolved `project_root` (never framework-relative).

## Verification (evidence required)
- Provide at least:
  - one command demonstrating `init` in a new empty directory
  - one command demonstrating idempotency / no overwrite behavior

## Out of scope
- Project root detection (Spec A)
- CLI entry point registration (Spec B)

