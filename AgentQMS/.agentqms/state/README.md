# AgentQMS State Files

This directory contains **runtime-generated snapshots** for debugging and audit purposes.

## ⚠️ Important

These are **NOT configuration inputs** and are **gitignored** (except this README).

### What These Files Are

- **`plugins.yaml`**: Snapshot of plugin discovery and loading at runtime (22 plugins)
- **`effective.yaml`**: Resolved configuration state snapshot

### What They Contain

Each snapshot captures:
- Discovery paths used during initialization
- Plugins found and loaded by type (artifact types, context bundles, validators)
- Full plugin metadata
- Validation errors (if any)
- Complete resolved plugin data

### Why They're Generated

The snapshots serve as:
1. **Debugging aids**: Understand what plugins were discovered/loaded
2. **Audit trails**: Track plugin discovery across different runs
3. **Configuration verification**: Validate resolved state after Phase 7+ fixes

### Current State (Phase 7+)

**Plugin System Status**: ✅ Operational
- 7 artifact types loaded (assessment, audit, bug_report, design_document, implementation_plan, vlm_report, walkthrough)
- 14 context bundles loaded
- Validators operational
- JSON schema validation temporarily disabled (artifact type validation via YAML rules)

**Registry**: Auto-generated from `AgentQMS/specs/` → `AgentQMS/.agentqms/registry.yaml`

### Why They Change Between Environments

These files contain **relative paths** from project root:
- Local dev: `AgentQMS/.agentqms/plugins` → relative path is portable
- CI/CD: Same relative path works in any checkout location
- Deployment: Works regardless of installation directory

### Gitignore Configuration

**Current setup** (post-Phase 7+):
```gitignore
# AgentQMS generated state files
AgentQMS/.agentqms/state/
!AgentQMS/.agentqms/state/README.md  # Exception: keep README
```

**Why**:
- Generated fresh on every plugin system initialization
- Contain only environment verification data
- Similar to `__pycache__/`, `node_modules/`, `.pytest_cache/`
- No functional impact if deleted

### Regenerating Snapshots

Snapshots are automatically regenerated when:
- Plugin system initializes
- `aqms plugin validate` runs
- Any tool using the plugin system executes

Manual regeneration:
```bash
uv run python -m AgentQMS.tools.core.plugins --validate --write-snapshot
```

### Reading the Snapshots

For debugging, inspect `plugins.yaml`:

```bash
cat AgentQMS/.agentqms/state/plugins.yaml | less
```

Key sections:
- `discovery_paths`: Where plugins were found
- `plugins_loaded`: Summary of loaded plugins by type
- `plugin_metadata`: Details of each loaded plugin
- `validation_errors`: Any issues during loading (should be empty post-Phase 7.1)
- `resolved`: Final resolved plugin configurations

### Middleware Observability (Phase C)

For middleware state and metrics, see:
- **Logs**: `outputs/logs/middleware/`
- **Metrics**: `outputs/metrics/`
- **Commands**: `make qms-middleware-health`, `make qms-middleware-dashboard`

---

**Last Updated**: 2026-02-02
**Schema Version**: 2.0 (Phase 7+)
**Plugin Status**: ✅ 22 plugins operational
