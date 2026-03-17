# Constitution — Global Stateless AgentQMS

## Non-negotiables
- All planning + session artifacts MUST live under `specs/global-agentqms/` (no workspace clutter).
- Session handovers use **Option A**: multiple files with timestamped filenames.

## Invariants (must always hold)
### Project root resolution
Order of precedence for `project_root` detection:
1. `AGENTQMS_PROJECT_ROOT` environment variable (absolute or relative, resolved to absolute).
2. Upward traversal from current working directory (CWD) to find a marker directory/file:
   - `.agentqms/` OR `AGENTS.yaml` (primary markers)
   - optionally `.git/` or `pyproject.toml` (secondary markers) if the implementation chooses to include them
3. Fallback: CWD (supports running in empty directories for `init`).

### Statelessness boundary
- The framework install location (where `AgentQMS` code lives) MUST NOT influence `project_root`.
- All project-scoped state MUST be confined to the resolved `project_root` (e.g., `.agentqms/` contents).

### Isolation & safety
- No cross-project registry contamination: resolved `project_root` defines the only writable scope.
- Global invocation MUST be deterministic: identical `AGENTQMS_PROJECT_ROOT` + CWD markers ⇒ identical root resolution.

## Evidence-first workflow
- Any spec completion must include at least one reproducible verification command + expected observable outcome.

