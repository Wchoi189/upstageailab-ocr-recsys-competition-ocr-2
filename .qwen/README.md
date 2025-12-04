# Qwen Coder + AgentQMS

## Table of Contents

- [Quick Reference](#quick-reference)
- [Chat Usage](#chat-usage)
- [Commands](#commands)
- [Configuration](#configuration)
- [Workflows](#workflows)

## Quick Reference

- **Agent Name:** `Qwen AgentQMS` (use `@Qwen AgentQMS` in chat)
- **Commands:** `.qwen/run.sh validate|create|interactive`
- **Config:** `.qwen/settings.json`
- **SST:** `AgentQMS/knowledge/agent/system.md`

## Chat Usage

### Agent Reference
- `@Qwen AgentQMS` - Full agent name in Cursor chat
- `@Qwen` - Short reference (if unique)

### Examples
- `@Qwen AgentQMS, validate all artifacts`
- `@Qwen AgentQMS, create bug report for auth issue`
- `@Qwen AgentQMS, check compliance and apply fixes`

### Command Delegation
- Reference commands: `Run: ./.qwen/run.sh validate`
- Delegate tasks: `@Qwen AgentQMS, handle validation`

## Commands

### Validation
```bash
./.qwen/run.sh validate
```

### Artifact Creation
```bash
./.qwen/run.sh create <type> <name> <title>
```
**Types:** plan, assessment, bug-report, design, research, template

### Interactive
```bash
./.qwen/run.sh interactive "<prompt>"
```

### Direct Qwen CLI
```bash
qwen --approval-mode yolo --include-directories /workspaces/upstageailab-ocr-recsys-competition-ocr-2 --prompt "<task>"
```

## Configuration

**Status:** Checkpointing disabled in `settings.json` (required fix)

**Key Settings:**
- Checkpointing: `enabled: false`
- Workspace: `/workspaces/upstageailab-ocr-recsys-competition-ocr-2`
- Model: `qwen/qwen3-coder`

## Workflows

### Validation
- Run: `./.qwen/run.sh validate`
- Chat: `@Qwen AgentQMS, validate artifacts`

### Artifact Creation
- Run: `./.qwen/run.sh create plan my-feature "Title"`
- Chat: `@Qwen AgentQMS, create plan named "my-feature" with title "Title"`

### Fixes
- Run: `./.qwen/run.sh validate && cd AgentQMS/interface && make fix`
- Chat: `@Qwen AgentQMS, apply fixes`

## Files

- `settings.json` - Qwen config (checkpointing disabled)
- `run.sh` - Wrapper script for AgentQMS operations
- `prompts.md` - Pre-built prompts for common tasks

## Notes

- Always reference AgentQMS context: `AgentQMS/knowledge/agent/system.md`
- Use automation only; never create artifacts manually
- Validate after changes: `make validate && make compliance`
