## AgentQMS – Copilot Instructions

This project uses the **AgentQMS** framework for quality management. Copilot should always route quality-related work through AgentQMS instead of ad‑hoc files.

### 1. First files to read

1. `AgentQMS/knowledge/agent/system.md` – **Single Source of Truth** (rules, do/don’t, workflows).
2. `.agentqms/state/architecture.yaml` – Component map, capabilities, and tool locations.
3. `.copilot/context/tool-catalog.md` – Available automation tools and Make targets.
4. `.copilot/context/workflow-triggers.yaml` – Task → workflow mapping.
5. `.copilot/context/context-bundles-index.md` – Context bundles for focused work.

### 2. How to discover tools and workflows

Run all AgentQMS commands from `AgentQMS/interface/`:

```bash
cd AgentQMS/interface

# Discovery & status
make help       # Show all available commands
make discover   # List available tools
make status     # Framework status check
```

### 3. How to create artifacts

Artifacts are **never** created manually. Always use Makefile targets:

```bash
cd AgentQMS/interface

make create-plan NAME=my-plan TITLE="My Plan"
make create-assessment NAME=my-assessment TITLE="My Assessment"
make create-audit NAME=my-audit TITLE="My Audit"
make create-bug-report NAME=my-bug TITLE="Bug Description"
```

- Artifacts live in `docs/artifacts/`.
- File names must follow: `YYYY-MM-DD_HHMM_[type]_name.md` (UTC timestamps recommended).

### 4. Validation and compliance

After artifacts or documentation are changed, always run:

```bash
cd AgentQMS/interface
make validate      # Validate all artifacts
make compliance    # Full compliance check
make boundary      # Boundary validation (framework vs project)
```

Fix any reported issues **before** merging changes.

### 5. Context loading for smarter assistance

For focused work, Copilot should load context instead of scanning the repo blindly:

```bash
cd AgentQMS/interface

make context TASK="short task description"
make context-development
make context-docs
make context-debug
make context-plan
```

Prefer these context bundles when proposing large edits, audits, or refactors.

### 6. Quality management rules (summary)

1. Check for an existing **implementation plan** before major work.
2. If none exists, create one with `make create-plan` **before** coding.
3. Document bugs using `make create-bug-report`, not ad‑hoc notes.
4. Use plugin-based extensions from `.agentqms/plugins/` instead of custom one‑off scripts.
5. Always finish by running `make validate` and `make compliance`.

For anything not covered here, defer to `AgentQMS/knowledge/agent/system.md`.
