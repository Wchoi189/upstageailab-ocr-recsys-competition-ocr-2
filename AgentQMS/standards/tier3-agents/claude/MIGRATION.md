# Claude Code Configuration Migration

## ⚠️ Directory Structure Change

The Claude Code configuration directory has been migrated:

**Old (Deprecated)**: `.ai-instructions/tier3-agents/claude/`
**New (Standard)**: `AgentQMS/standards/tier3-agents/claude/`

## What Was Migrated

### Configuration File
- **File**: `.claude.json` (MCP server configuration)
- **Old location**: `.ai-instructions/tier3-agents/claude/.claude.json`
- **New location**: `AgentQMS/standards/tier3-agents/claude/.claude.json`
- **Compatibility**: Symlink created from old → new location

### Runtime Data (Unchanged)
The following runtime data remains in `.ai-instructions/tier3-agents/claude/`:
- `history.jsonl` - Command history
- `projects/` - Project-specific sessions
- `shell-snapshots/` - Shell state snapshots
- `statsig/` - Analytics data
- `todos/` - Todo lists
- `debug/` - Debug logs

> **Note**: Runtime data stays in `.ai-instructions` as it's ephemeral and not part of the project standards.

## Why This Change?

1. **Standardization**: Configuration belongs in `AgentQMS/standards/` with other agent configs
2. **Version Control**: Config should be committed, runtime data should not
3. **Consistency**: Aligns with other tier3 agents (copilot, cursor, gemini)

## Impact

### ✅ No Breaking Changes
- Symlink ensures backward compatibility
- Claude Code will continue to work without changes
- Existing sessions and history preserved

### 📝 Updated References
All documentation references have been updated to point to the new location.

## File Locations

```
AgentQMS/standards/tier3-agents/claude/
├── .claude.json          # MCP server config (NEW LOCATION)
├── config.yaml           # Agent configuration
├── quick-reference.yaml  # Quick reference guide
├── settings.local.json   # Local settings
└── validation.sh         # Validation script

.ai-instructions/tier3-agents/claude/
├── .claude.json          # Symlink → AgentQMS/standards/...
├── history.jsonl         # Runtime: command history
├── projects/             # Runtime: project sessions
├── shell-snapshots/      # Runtime: shell states
├── statsig/              # Runtime: analytics
├── todos/                # Runtime: todo lists
└── debug/                # Runtime: debug logs
```

## For Developers

### Updating MCP Configuration
Edit the file at the **new location**:
```bash
code AgentQMS/standards/tier3-agents/claude/.claude.json
```

### Checking Configuration
```bash
# Both paths work due to symlink
cat .ai-instructions/tier3-agents/claude/.claude.json
cat AgentQMS/standards/tier3-agents/claude/.claude.json
```

### Adding New MCP Servers
Use the CLI (automatically updates the correct file):
```bash
claude mcp add <server-name> -- <command>
```

## Migration Complete ✅

All configuration has been migrated to the new standard location while maintaining full backward compatibility.
