---
title: "Tracking CLI Quick Reference"
date: "2025-11-06 00:00 (KST)"
type: "guide"
category: "ai_agent"
status: "active"
version: "1.0"
tags: ["reference", "agent-tools", "tracking", "docs"]
---

Tracking CLI Quick Reference
============================

Ultra-concise commands for the tracking CLI (`AgentQMS/agent_tools/utilities/tracking/cli.py`). For tool catalog and governance workflows, see `AgentQMS/knowledge/agent/tool_catalog.md`.

Quick setup
-----------
- Run from `agent/` unless noted.
- Ensure the DB exists: `make track-init`

- Create a plan / experiment
```bash
python AgentQMS/agent_tools/utilities/tracking/cli.py plan new --title "My Feature" --owner me
python AgentQMS/agent_tools/utilities/tracking/cli.py exp new --title "Prompt tuning" --objective "Reduce tokens"
```

- Add run / summarize experiment
```bash
python AgentQMS/agent_tools/utilities/tracking/cli.py exp run-add <key> 1 --params '{"k":1}' --outcome inconclusive
python AgentQMS/agent_tools/utilities/tracking/cli.py exp summarize <key> --style short --points "Tokens -15%"
```

- Status and CSV export
```bash
python AgentQMS/agent_tools/utilities/tracking/cli.py plan status --concise
make exp-export OUT=data/ops/experiment_runs.csv
```

What to track
-------------
- **Plans**: manage feature work; ensure tasks complete before marking done.
- **Refactors**: track cleanup/migration efforts and enforce closure.
- **Debug sessions**: capture hypothesis, scope, and running notes.
- **Experiments**: record runs/artifacts; summarize results for reuse.
- **Dashboards**: Streamlit Tracking Dashboard reads the same DB for status.

Status recipes
-------------
- Concise overview:
```bash
python -c "from AgentQMS.agent_tools.utilities.tracking.query import get_status; print(get_status('all'))"
```
- Plans only:
```bash
python -c "from AgentQMS.agent_tools.utilities.tracking.query import get_status; print(get_status('plan'))"
```
- Experiments with latest run outcome:
```bash
python -c "from AgentQMS.agent_tools.utilities.tracking.query import get_status; print(get_status('experiment'))"
```

Data export
-----------
```bash
make exp-export OUT=data/ops/experiment_runs.csv
```

Troubleshooting
--------------
- If commands fail with `no such table`, run `make track-init` to bootstrap the schema.
- Ensure you're in `agent/` so relative paths resolve.
- The Streamlit Tracking Dashboard uses the same DB—if it shows empty data, the CLI likely needs `make track-init` too.
- Need tooling beyond the tracking CLI? See `../automation/tooling_overview.md` and the generated `../automation/tool_catalog.md`.


