# Legacy Agent Configurations (DEPRECATED)

**Status**: Archived
**Deprecation Date**: 2025-12-16
**Retention Period**: 30 days (delete after 2026-01-15)
**Migration Target**: `.ai-instructions/tier3-agents/`

## Archived Directories

### `.claude/`
**Reason**: Superseded by `.ai-instructions/tier3-agents/claude/`
**Migration**: All configurations migrated to ADS v1.0 YAML format
**References**: Update any imports to `.ai-instructions/tier3-agents/claude/config.yaml`

### `.copilot/`
**Reason**: Superseded by `.ai-instructions/tier3-agents/copilot/`
**Migration**:
- `context/tool-catalog.md` → `.ai-instructions/tier2-framework/tool-catalog.yaml`
- `context/agentqms-overview.md` → `.ai-instructions/tier3-agents/copilot/config.yaml`
**References**: Update `.github/copilot-instructions.md` (already updated)

### `.cursor/`
**Reason**: Superseded by `.ai-instructions/tier3-agents/cursor/`
**Migration**: All configurations migrated to ADS v1.0 YAML format
**References**: Update any Cursor IDE references

### `.gemini/`
**Reason**: Superseded by `.ai-instructions/tier3-agents/gemini/`
**Migration**: All configurations migrated to ADS v1.0 YAML format
**References**: Update any Gemini API integration references

## Migration Notes

**Token Reduction**: Legacy configs totaled ~8,500 lines. New configs: 531 lines (94% reduction)
**Compliance**: All new configs validate against ADS v1.0 specification
**Self-Healing**: Pre-commit hooks enforce naming/placement/compliance automatically

## Action Required

If you encounter references to these directories:
1. Check `.ai-instructions/tier3-agents/{agent}/config.yaml` for equivalent
2. Update imports to new locations
3. Use `make discover` to find correct tool references

## Removal Instructions

After 2026-01-15:
```bash
rm -rf .ai-instructions/DEPRECATED/legacy-agent-configs
```
