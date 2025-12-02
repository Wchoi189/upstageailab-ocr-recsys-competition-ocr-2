# AgentQMS Migration - Status Update

## ✅ Completed

1. **Priority Components:**
   - ✅ Data contract schema → `AgentQMS/conventions/schemas/data_contract.json`
   - ✅ OCR directory → `AgentQMS/ocr/` (containerized)
   - ✅ VLM directory verified (at `agent_qms/vlm/`)

2. **Framework Setup:**
   - ✅ `AgentQMS/__init__.py` created
   - ✅ `.agentqms/settings.yaml` configured
   - ✅ `.agentqms/plugins/` structure created
   - ✅ `.agentqms/state/` structure created
   - ✅ GitHub workflows merged

3. **Migration Scripts:**
   - ✅ `finalize_migration.py` - Ready to execute
   - ✅ `move_framework_simple.py` - Alternative script
   - ✅ `migrate_agentqms.py` - Enhanced migration script

## 🔄 Pending: Execute Directory Moves

The following directories need to be **moved** from `__NEW__/` to project root. Scripts have been prepared but need execution:

### Execute This Command:

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2
python3 finalize_migration.py
```

This will move:
- `agent_tools/` → `AgentQMS/agent_tools/`
- `interface/` → `AgentQMS/interface/`
- `knowledge/` → `AgentQMS/knowledge/`
- `toolkit/` → `AgentQMS/toolkit/`
- `CHANGELOG.md` → `AgentQMS/CHANGELOG.md`
- `conventions/audit_framework/` → `AgentQMS/conventions/audit_framework/`
- `conventions/templates/` → `AgentQMS/conventions/templates/`
- `conventions/q-manifest.yaml` → `AgentQMS/conventions/q-manifest.yaml`
- `.copilot/` → `.copilot/` (or merge)
- `.qwen/` → `.qwen/` (or merge)
- `.cursor/` → `.cursor/` (or merge)

### Manual Alternative:

If the script doesn't work, use these commands manually:

```bash
cd /workspaces/upstageailab-ocr-recsys-competition-ocr-2

# Move AgentQMS components
mv __NEW__/AgentQMS/agent_tools AgentQMS/
mv __NEW__/AgentQMS/interface AgentQMS/
mv __NEW__/AgentQMS/knowledge AgentQMS/
mv __NEW__/AgentQMS/toolkit AgentQMS/
mv __NEW__/AgentQMS/CHANGELOG.md AgentQMS/

# Move conventions items
mv __NEW__/AgentQMS/conventions/audit_framework AgentQMS/conventions/
mv __NEW__/AgentQMS/conventions/templates AgentQMS/conventions/
mv __NEW__/AgentQMS/conventions/q-manifest.yaml AgentQMS/conventions/

# Handle dot directories (merge if exist)
if [ -d .copilot ]; then
    cp -rn __NEW__/.copilot/* .copilot/
    rm -rf __NEW__/.copilot
else
    mv __NEW__/.copilot .
fi

if [ -d .qwen ]; then
    cp -rn __NEW__/.qwen/* .qwen/
    rm -rf __NEW__/.qwen
else
    mv __NEW__/.qwen .
fi

if [ -d .cursor ]; then
    cp -rn __NEW__/.cursor/* .cursor/
    rm -rf __NEW__/.cursor
else
    mv __NEW__/.cursor .
fi
```

## Verification

After moving, verify structure:
```bash
ls -la AgentQMS/
# Should show: agent_tools/, interface/, knowledge/, toolkit/, conventions/, ocr/, CHANGELOG.md, __init__.py
```

## Next Steps After Moves

1. Verify framework structure is complete
2. Run validation: `cd AgentQMS/interface && make validate`
3. Test imports
4. Clean up `__OLD__` directory (optional, after verification)
