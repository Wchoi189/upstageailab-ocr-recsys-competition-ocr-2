---
title: "Agent Tooling Overview"
date: "2025-11-06"
type: "guide"
category: "ai_agent"
status: "active"
version: "1.0"
tags: ["agent-tools", "tracking", "artifacts"]
---

Agent Tooling Overview
======================

High-level map of automation available under `scripts/agent_tools/`. For full inventory see the generated `tool_catalog.md`; for tracking command details use `../tracking/cli_reference.md`.

Core workflows
--------------
- **Artifact creation**: `python scripts/agent_tools/core/artifact_workflow.py create --type implementation_plan --name my-plan --title "My Plan"`
- **Validation suite**: `python scripts/agent_tools/compliance/validate_artifacts.py --all`
- **Index maintenance**: `python scripts/agent_tools/documentation/update_artifact_indexes.py --all`
- **Tool discovery**: `python scripts/agent_tools/core/discover.py`

Tracking Utilities (SQLite-backed)
----------------------------------
- Location: `scripts/agent_tools/utilities/tracking/`
- Initialize DB: `make -C agent track-init`
- CLI examples:
  - Plans: `python scripts/agent_tools/utilities/tracking/cli.py plan status --concise`
  - Debug: `python scripts/agent_tools/utilities/tracking/cli.py debug new --title "Latency spike"`
  - Experiments: `python scripts/agent_tools/utilities/tracking/cli.py exp summarize <key> --style short --points "Tokens -15%"`
- Export runs: `make -C agent exp-export OUT=data/ops/experiment_runs.csv`
- Dashboard: `streamlit_app/pages/10_Tracking_Dashboard.py`

Artifact Workflow Integration
-----------------------------
- All artifact creation goes through `scripts/agent_tools/core/artifact_workflow.py`.
- **Implementation Plans**: Always use Blueprint Protocol Template (PROTO-GOV-003). Generator uses this automatically.
- Tracking CLI hooks:
  - `debug new` → creates `bug_report` artifact (minimal)
  - `exp summarize` → creates `research` artifact and links it

More references
---------------
- Generated inventory → `tool_catalog.md`
- Tracking commands → `../tracking/cli_reference.md`
- Changelog automation → `changelog_process.md`
