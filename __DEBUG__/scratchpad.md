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
  --ignore 'AgentQMS/tests,AgentQMS/standards/tier3-agents/*,AgentQMS/.mcp-telemetry.jsonl,*.bak,AgentQMS/bin/artifacts_violations_history.json,AgentQMS/bin/cli_tools/audio,AgentQMS/mcp_schema.yaml,AgentQMS/context-tooling-2.0-plan.md,AgentQMS/.archive,*.py,*.pyc,AgentQMS/mcp_server.py' \
  --output /workspaces/upstageailab-ocr-recsys-competition-ocr-2/AgentQMS_2026-01-27.md

```



### Commonly used OCR module

```bash
repomix --style markdown \
  --include 'ocr/core/infrastructure' \
  --ignore '' \
  --output /workspaces/upstageailab-ocr-recsys-competition-ocr-2/ocr_infrastructure_for_multi-agent_2026-01-23.md

```


### Quick GPU Test (Recommended)
```bash
uv run python scripts/runners/train.py \
  domain=recognition \
  experiment=rec_baseline_official \
  trainer.max_epochs=10 \
  data.batch_size=64 \
  dataloaders.train_dataloader.num_workers=4 \
  dataloaders.val_dataloader.num_workers=4 \
  dataloaders.train_dataloader.pin_memory=true \
  dataloaders.val_dataloader.pin_memory=true \
  +dataloaders.train_dataloader.batch_size=64 \
  +dataloaders.val_dataloader.batch_size=64 \
  +train/logger=wandb
```


```bash
uv run python scripts/runners/train.py \
  domain=recognition \
  ckpt_path="outputs/checkpoints/last.ckpt" \
  experiment=rec_baseline_official \
  trainer.max_epochs=10 \
  +train/logger=wandb
```


```bash
uv run python scripts/runners/train.py \
  domain=recognition \
  ckpt_path="outputs/checkpoints/last.ckpt" \
  experiment=rec_baseline_official \
  trainer.max_epochs=1 \
  +train/logger=wandb
```
