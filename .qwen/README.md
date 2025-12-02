# Qwen Coder + AgentQMS Integration

This directory contains configuration and scripts for running Qwen Coder with AgentQMS framework integration.

## ✅ Current Status: FIXED

**Issue Resolved**: Qwen CLI checkpointing bug was fixed by disabling checkpointing in settings.json.
**Status**: Fully functional with AgentQMS integration.

## Files

- `settings.json` - Qwen configuration optimized for AgentQMS (checkpointing disabled)
- `prompts.md` - Pre-built prompts for common AgentQMS tasks
- `run.sh` - Convenience script for AgentQMS operations
- `manual_validate.sh` - Manual document validation script (backup)

## ✅ Working Usage

### Document Validation
```bash
./.qwen/run.sh validate
# Generates a prompt for AI-assisted document validation
```

### Create New Artifacts
```bash
./.qwen/run.sh create plan my-feature "Feature Plan"
./.qwen/run.sh create assessment security-review "Security Assessment"
./.qwen/run.sh create bug-report auth-issue "Authentication Bug Report"
```

### Direct Qwen Commands
```bash
qwen --approval-mode yolo --include-directories /workspaces/upstageailab-ocr-recsys-competition-ocr-2 --prompt "Follow AgentQMS/knowledge/agent/system.md and [your task]"
```

### Manual Validation (Backup)
```bash
./.qwen/run.sh validate-manual
```

## Manual Validation Results

The manual validation script found **27 files** with naming convention violations in `docs/artifacts/`. All files need:

1. **Renaming** to format: `YYYY-MM-DD_HHMM_{ARTIFACT_TYPE}_descriptive-name.md`
2. **Frontmatter** addition with proper YAML headers
3. **Directory** restructuring according to artifact categories

Valid artifact types: `implementation_plan_`, `assessment-`, `audit-`, `design-`, `research-`, `template-`, `BUG_`, `SESSION_`

## AgentQMS Integration

- **Instructions**: Points to `AgentQMS/knowledge/agent/system.md`
- **Architecture**: References `.agentqms/state/architecture.yaml`
- **Artifacts Path**: `docs/artifacts/`
- **Validation**: Configured to ignore non-artifact files

## Configuration

**Key Fix**: Checkpointing disabled in `settings.json`:
```json
{
  "general": {
    "checkpointing": {
      "enabled": false
    }
  }
}
```

## Next Steps

1. **Run validation**: `./.qwen/run.sh validate`
2. **Fix document issues** using the generated prompts
3. **Create new artifacts** with the create commands
4. **Use direct Qwen commands** for custom tasks

## For Other AI Tools

The generated prompts work with any AI tool:
- ChatGPT
- Claude
- Other AI coding assistants

The AgentQMS framework ensures consistent development practices across all tools!
- Adjust AgentQMS paths
- Configure different models

Edit `prompts.md` to add new task templates.
