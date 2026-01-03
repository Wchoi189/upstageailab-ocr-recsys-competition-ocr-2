# Claude Code MCP Servers - Quick Reference

## ✅ Successfully Configured MCP Servers

All 3 MCP servers are now active in Claude Code:

1. **project_compass** - ✓ Connected
2. **agentqms** - ✓ Connected
3. **experiments** - ✓ Connected

## Using MCP Servers in Claude Code

### List Available Servers
```bash
claude mcp list
```

### View Server Tools
Start a Claude Code session and the MCP tools will be automatically available.

### Available MCP Tools

#### project_compass
- `list_resources` - List all available project resources
- `read_resource` - Read project state files (compass.json, session_handover.md, etc.)

#### agentqms
- `create_artifact` - Create new artifacts (assessments, walkthroughs, etc.)
- `validate_artifact` - Validate artifact compliance
- `list_artifact_templates` - List available templates
- `get_standard` - Retrieve project standards
- `check_compliance` - Check overall compliance

#### experiments
- `init_experiment` - Initialize new experiment
- `add_task` - Add task to experiment
- `log_insight` - Log insights/decisions/failures
- `get_experiment_status` - Get experiment status
- `sync_experiment` - Sync experiment to database

## Configuration Files

- **MCP Config**: `.ai-instructions/tier3-agents/claude/.claude.json`
- **Wrapper Scripts**: `.devcontainer/scripts/mcp-*.sh`
- **MCP Template**: `.devcontainer/mcp_config.json`

## Troubleshooting

### Check Server Status
```bash
claude mcp list
```

### Test Individual Server
```bash
bash .devcontainer/scripts/mcp-project-compass.sh
# Press Ctrl+C to stop
```

### Remove a Server
```bash
claude mcp remove <server-name>
```

### Re-add a Server
```bash
claude mcp add --transport stdio <server-name> -- bash /path/to/wrapper.sh
```

## Next Steps

1. Start Claude Code: `claude`
2. MCP servers will auto-connect
3. Use MCP tools in your prompts
4. Example: "List all project compass resources"

---

**All set!** Your MCP servers are ready to use in Claude Code! 🎉
