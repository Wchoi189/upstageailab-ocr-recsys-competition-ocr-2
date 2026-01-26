## Notes





### Commonly used Repomix CLI

```bash
repomix --style markdown \
  --include 'AgentQMS/*,AgentQMS/standards/tier3-agents/multi-agent-system.yaml' \
  --ignore 'AgentQMS/standards/tier3-agents/*,*.jsonl,*.bak,AgentQMS/bin/artifacts_violations_history.json,AgentQMS/bin/cli_tools/audio,AgentQMS/mcp_server.py,AgentQMS/mcp_schema.yaml,AgentQMS/context-tooling-2.0-plan.md,' \
  --output /workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS_2026-01-26.md

```

```bash
repomix --style markdown \
  --include 'AgentQMS/' \
  --ignore 'AgentQMS/standards/tier3-agents/*,*.jsonl,*.bak,AgentQMS/bin/artifacts_violations_history.json,AgentQMS/bin/cli_tools/audio,AgentQMS/mcp_server.py,AgentQMS/mcp_schema.yaml,AgentQMS/context-tooling-2.0-plan.md,AgentQMS/.archive,*.py' \
  --output /workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS_2026-01-27.md

```



### Commonly used OCR module

```bash
repomix --style markdown \
  --include 'ocr/core/infrastructure' \
  --ignore '' \
  --output /workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr_infrastructure_for_multi-agent_2026-01-23.md

```


