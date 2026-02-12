# MCP Resources Configuration Update - V3.0.1

**Date:** 2026-02-12
**Context:** Updated MCP server resources to reflect V3 architecture

---

## File Updated

### [scripts/mcp/config/resources.yaml](../../scripts/mcp/config/resources.yaml) ✅

**Changes Made:**

#### Removed (V2 Legacy Resources)
```yaml
# ❌ Removed - No longer exists in V3
- compass://compass.json              → Replaced by vessel_state.json
- compass://session_handover.md       → V3 doesn't use session handovers
- compass://current_session.yml       → Replaced by vessel state
- compass://uv_lock_state.yml         → Legacy (kept in comments for reference)
```

#### Added (V3 Resources)
```yaml
# ✅ Added - V3 Architecture
- compass://vessel_state.json         → .vessel/vessel_state.json (V3 state)
- compass://agents.yaml               → AGENTS.yaml (updated)
- compass://agents.md                 → AGENTS.md (V3 documentation)
- compass://changelog.md              → CHANGELOG.md (version history)
```

---

## Architecture Migration: V2 → V3

### V2 State Management (Removed)
```
compass.json                    # Legacy state file
session_handover.md             # Session handover document
active_context/
  └── current_session.yml       # Active session metadata
```

**Issues:**
- Multiple state files (fragmented)
- Session handover pattern (complex)
- Manual JSON editing required

### V3 State Management (Current)
```
.vessel/
  └── vessel_state.json         # Single source of truth
```

**Benefits:**
- Single state file
- Skills-based interface (no manual editing)
- Auto-managed by CLI/Skills
- Vault directives auto-injected

---

## Resource Mappings

| V2 Resource | V3 Resource | Status | Notes |
|-------------|-------------|--------|-------|
| `compass://compass.json` | `compass://vessel_state.json` | ✅ Migrated | New location: `.vessel/` |
| `compass://session_handover.md` | N/A | ❌ Removed | V3 uses vault directives |
| `compass://current_session.yml` | `compass://vessel_state.json` | ✅ Merged | Integrated into vessel state |
| `compass://uv_lock_state.yml` | N/A | ⚠️ Legacy | Kept for reference only |
| `compass://agents.yaml` | `compass://agents.yaml` | ✅ Updated | Paths corrected |
| N/A | `compass://agents.md` | ✅ New | V3 documentation |
| N/A | `compass://changelog.md` | ✅ New | Version history |

---

## Verification

### Resource Paths
```bash
# V3 Resources (all exist)
$ ls -la dev_tools/project_compass/.vessel/vessel_state.json
✅ -rw------- 1 vscode vscode 5058 Feb 12 20:10

$ ls -la dev_tools/project_compass/AGENTS.yaml
✅ -rw-rw-r-- 1 vscode vscode 1952 Feb 12 21:18

$ ls -la dev_tools/project_compass/AGENTS.md
✅ -rw-rw-r-- 1 vscode vscode 8277 Feb 12 21:18

$ ls -la dev_tools/project_compass/CHANGELOG.md
✅ -rw-rw-r-- 1 vscode vscode 7317 Feb 12 21:19
```

### V2 Resources (removed)
```bash
$ ls -la dev_tools/project_compass/compass.json
❌ No such file or directory

$ ls -la dev_tools/project_compass/session_handover.md
❌ No such file or directory

$ ls -la dev_tools/project_compass/active_context/
❌ No such file or directory
```

---

## Impact Assessment

### Unified MCP Server
- **Status:** ✅ Updated
- **Impact:** Resources now point to correct V3 files
- **Breaking Change:** Yes - clients using old URIs will need to update

### Deprecated URIs
Clients using these URIs will receive errors:
- `compass://compass.json` → Use `compass://vessel_state.json`
- `compass://session_handover.md` → Not available in V3
- `compass://current_session.yml` → Use `compass://vessel_state.json`

### Migration Path for Clients
```yaml
# Before (V2)
resource = await read_resource("compass://compass.json")

# After (V3)
resource = await read_resource("compass://vessel_state.json")
```

---

## Related Files

### Also Updated
1. [resources.yaml](../../scripts/mcp/config/resources.yaml) - Resource definitions
2. [AGENTS.yaml](AGENTS.yaml) - Paths corrected (v3.0.1)
3. [AGENTS.md](AGENTS.md) - Directory structure updated (v3.0.1)
4. Skills SKILL.md files - Paths corrected (v3.0.1)

### Not Changed (Correct)
1. [unified_server.py](../../scripts/mcp/unified_server.py) - No hardcoded paths
2. [sync_configs.py](../../scripts/mcp/sync_configs.py) - Config sync script

---

## Testing Checklist

- [x] Verify all V3 resource paths exist
- [x] Confirm V2 resources removed
- [x] Update resource URIs in config
- [ ] Test MCP server with new resources (requires server restart)
- [ ] Verify client access to vessel_state.json
- [ ] Update client code to use new URIs

---

## Documentation Comments

Added to resources.yaml:
```yaml
# V2 Resources Removed:
# - compass.json -> Replaced by .vessel/vessel_state.json
# - session_handover.md -> Removed (V3 doesn't use session handovers)
# - active_context/current_session.yml -> Removed (replaced by vessel state)
# - environments/uv_lock_state.yml -> Kept for reference (legacy)
```

---

**Update Status:** ✅ COMPLETE
**Breaking Changes:** YES - V2 URIs deprecated
**Migration Required:** YES - Update client code to use new URIs
**Documentation:** COMPLETE

---

Last Updated: 2026-02-12
By: Post-Nuclear Architecture Audit
Session: Documentation sync for V3.0.1
