# .qwen/ Directory - AgentQMS Integration

**Last Updated**: 2025-12-06 22:30 (KST)  
**Status**: Active - Organized schema established

## Purpose

This directory contains Qwen AI agent integration files for AgentQMS. All tools here are **deprecated wrappers** - use AgentQMS tools directly instead.

## Directory Schema

```
.qwen/
├── README.md              # This file - directory schema and guidance
├── QWEN.md                # Agent context and quick reference (ACTIVE)
├── settings.json          # Qwen CLI settings (deprecated - see note)
└── archive/               # Deprecated/obsolete files
    ├── python/           # Old Python scripts (replaced by AgentQMS)
    ├── shell/            # Old shell scripts (replaced by Makefile)
    ├── README_old.md     # Original README (pre-schema)
    ├── README_new.md     # Attempted update (stale references)
    └── prompts.md        # Qwen-specific prompts
```

## Active Files

### QWEN.md (Agent Context)
- **Purpose**: Quick reference for AI agents using AgentQMS
- **Audience**: AI agents (Claude, ChatGPT, etc.)
- **Content**: Current AgentQMS structure, commands, workflows
- **Maintenance**: Update when AgentQMS structure changes

### settings.json (Qwen CLI Config)
- **Purpose**: Configuration for Qwen CLI tool
- **Status**: Deprecated - Qwen CLI integration discontinued
- **Action**: Keep for historical reference, do not update

## Archived Files (Obsolete)

All files below are **obsolete** and replaced by AgentQMS tools:

### Python Scripts → Replaced by `AgentQMS/agent_tools/audit/artifact_audit.py`
- ❌ `fix_frontmatter.py` - Use `artifact_audit.py --batch N`
- ❌ `fix_batch1_batch2.py` - Use `artifact_audit.py --batch 1` or `--batch 2`
- ❌ `final_batch_fix.py` - Use `artifact_audit.py --all`

**Replacement:**
```bash
# Old way (obsolete)
python .qwen/fix_frontmatter.py --batch 2

# New way (current)
cd AgentQMS/interface
make audit-fix-batch BATCH=2
# OR
python AgentQMS/agent_tools/audit/artifact_audit.py --batch 2
```

### Shell Scripts → Replaced by `AgentQMS/interface/Makefile`
- ❌ `run.sh` - Echo-only wrapper, never executed Qwen properly
- ❌ `run_improved.sh` - Attempted Qwen execution but had checkpointing issues
- ❌ `manual_validate.sh` - Use `make validate` instead
- ❌ `qwen-chat.sh` - Qwen CLI integration discontinued

**Replacement:**
```bash
# Old way (obsolete)
bash .qwen/run.sh validate

# New way (current)
cd AgentQMS/interface
make validate
```

### Documentation → Consolidated into QWEN.md
- ❌ `README_old.md` - Original README (replaced by this schema)
- ❌ `README_new.md` - Contained stale references to consolidate.py
- ❌ `prompts.md` - Qwen-specific prompts, no longer relevant

## Migration Guide

### If You Find References to Obsolete Files

| Old Reference | New Replacement |
|---------------|-----------------|
| `.qwen/consolidate.py` | `AgentQMS/agent_tools/audit/artifact_audit.py` |
| `.qwen/fix_*.py` | `artifact_audit.py` with appropriate flags |
| `.qwen/run*.sh` | `cd AgentQMS/interface && make <target>` |
| `.qwen/manual_validate.sh` | `make validate` |
| `.qwen/README_new.md` | `.qwen/QWEN.md` |

### AgentQMS Commands Reference

**Audit & Fix:**
```bash
cd AgentQMS/interface
make audit-fix-batch BATCH=1         # Preview batch 1
make audit-fix-batch-apply BATCH=1   # Apply batch 1 fixes
make audit-fix-all                   # Fix all artifacts
make audit-report                    # Report violations
```

**Validation:**
```bash
cd AgentQMS/interface
make validate                        # Validate all artifacts
make compliance                      # Check compliance
make boundary                        # Check boundaries
```

**Creation:**
```bash
cd AgentQMS/interface
make create-plan NAME=my-plan TITLE="My Plan"
make create-assessment NAME=my-assessment TITLE="Assessment"
```

## Cleanup Status

### Archived (2025-12-06)

**Python Scripts** (→ `archive/python/`):
- `fix_frontmatter.py`
- `fix_batch1_batch2.py`
- `final_batch_fix.py`

**Shell Scripts** (→ `archive/shell/`):
- `run.sh`
- `run_improved.sh`
- `manual_validate.sh`
- `qwen-chat.sh`

**Documentation** (→ `archive/`):
- `README_old.md` (original)
- `README_new.md` (stale)
- `prompts.md` (Qwen-specific)

### Retained

- `QWEN.md` - Active agent context
- `settings.json` - Historical reference
- `README.md` - This schema document

## Notes for AI Agents

**⚠️ Important:** If you encounter references to:
- `.qwen/consolidate.py`
- `.qwen/fix_*.py`
- `.qwen/run*.sh`

These are **obsolete**. Use AgentQMS tools instead:
- Read: `AgentQMS/knowledge/agent/system.md`
- Reference: `.qwen/QWEN.md`
- Execute: `cd AgentQMS/interface && make <command>`

## Qwen CLI Integration (Deprecated)

**Status**: Discontinued due to:
1. Checkpointing issues with Git detection
2. Inconsistent command syntax across versions
3. Better integration via VS Code Copilot and native tools

**Replacement**: Use GitHub Copilot (Claude/GPT-4) with AgentQMS context bundles.

## Schema Maintenance

**Update Triggers:**
- New files added to `.qwen/`
- AgentQMS structure changes
- Tool deprecations or replacements
- Integration method changes

**Update Process:**
1. Document new file purpose and status
2. Move obsolete files to `archive/`
3. Update this README schema
4. Update `.qwen/QWEN.md` if agent workflows change

---

**Questions?** See `AgentQMS/knowledge/agent/system.md` for full AgentQMS documentation.
