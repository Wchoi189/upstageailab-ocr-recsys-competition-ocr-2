# AgentQMS Migration Status

## Completed ✅

1. **Priority Components Migrated:**
   - ✅ `data_contract.json` schema → `AgentQMS/conventions/schemas/data_contract.json`
   - ✅ OCR directory → `AgentQMS/ocr/` (containerized within framework)
   - ✅ VLM directory verified (already at `agent_qms/vlm/`)

2. **Framework Components:**
   - ✅ Created `AgentQMS/__init__.py`
   - ✅ Created `.agentqms/settings.yaml` with path configurations
   - ✅ Created `.agentqms/version`
   - ✅ Created `.agentqms/plugins/` directory structure with:
     - `validators.yaml`
     - `artifact_types/audit.yaml`
     - `artifact_types/change_request.yaml`
     - `context_bundles/security-review.yaml`
   - ✅ Created `.agentqms/state/` directory with:
     - `README.md`
     - `architecture.yaml`

3. **GitHub Workflows:**
   - ✅ Merged `agentqms-ci.yml`
   - ✅ Merged `agentqms-autofix.yml`
   - ✅ Merged `agentqms-validation.yml`

4. **Migration Scripts:**
   - ✅ Enhanced migration script: `__NEW__/AgentQMS/scripts/migrate_agentqms.py`
   - ✅ Framework move scripts created

## Remaining Tasks 🔄

### Critical: Move Framework Directories

The following large directories need to be **moved** (not copied) from `__NEW__/AgentQMS/` to `AgentQMS/`:

1. **AgentQMS Framework** (from `__NEW__/AgentQMS/`):
   - `agent_tools/` - Core implementation layer
   - `conventions/audit_framework/` - Audit framework
   - `conventions/templates/` - Artifact templates
   - `conventions/q-manifest.yaml` - Q-manifest
   - `interface/` - Agent interface layer
   - `knowledge/` - Protocols and references
   - `toolkit/` - Legacy compatibility shim
   - `CHANGELOG.md`

2. **Configuration Directories** (from `__NEW__/`):
   - `.copilot/` - Copilot context files (move or merge)
   - `.qwen/` - Qwen configuration (merge with existing)
   - `.cursor/` - Cursor configuration (merge with existing)

### Execution Commands

Use Python to move directories:
```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
python3 move_framework_simple.py
```

Or use manual terminal commands:
```bash
# Move AgentQMS framework directories
mv __NEW__/AgentQMS/agent_tools AgentQMS/
mv __NEW__/AgentQMS/interface AgentQMS/
mv __NEW__/AgentQMS/knowledge AgentQMS/
mv __NEW__/AgentQMS/toolkit AgentQMS/
mv __NEW__/AgentQMS/CHANGELOG.md AgentQMS/

# Move conventions subdirectories
mv __NEW__/AgentQMS/conventions/audit_framework AgentQMS/conventions/
mv __NEW__/AgentQMS/conventions/templates AgentQMS/conventions/
mv __NEW__/AgentQMS/conventions/q-manifest.yaml AgentQMS/conventions/

# Move/merge dot directories (if they don't exist, move; if they exist, merge)
# .copilot, .qwen, .cursor - handle based on existence
```

## Notes

- OCR directory has been moved to `AgentQMS/ocr/` to keep project-specific tools within the framework container
- Import statements verified - all use correct paths (`agent_qms.vlm` remains correct)
- Path configurations in `.agentqms/settings.yaml` are set for project root
- Migration scripts created but may need manual execution due to path/permission issues

## Next Steps

1. **Execute manual moves** using the commands above or run `move_framework_simple.py`
2. Verify all directories moved correctly
3. Test framework validation: `cd AgentQMS/interface && make validate`
4. Verify imports work correctly
5. Clean up `__OLD__` directory (after verification)
