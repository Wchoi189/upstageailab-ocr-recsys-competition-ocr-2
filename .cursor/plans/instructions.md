---
title: AgentQMS – Cursor Instructions
updated: 2025-11-27 15:20 UTC
---

1. Read `AgentQMS/knowledge/agent/system.md` (SST) before acting.
2. Use automation only: run `cd AgentQMS/interface && make help` for commands.
3. Artifacts live in `artifacts/` with names `YYYY-MM-DD_HHMM_[type]_name.md`.
4. Always validate after changes: `make validate && make compliance`.
5. Need context? run `make context` (auto-detects) or `context-development/docs/debug/plan`.
6. For architecture map + tools, open:
   - `.agentqms/state/architecture.yaml`
   - `.copilot/context/tool-catalog.md`
   - `.copilot/context/workflow-triggers.yaml`
7. Never edit artifacts manually; use `make create-*` targets.

Follow the SST for anything not covered here.