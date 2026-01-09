---
title: Context System Architecture Analysis - Fragmentation Assessment
date: 2026-01-09 15:30 (KST)
type: assessment
category: architecture
status: draft
version: '1.0'
ads_version: '1.0'
related_artifacts:
  - 2026-01-09_1515_research-agentqms-plugin-system-evolution.md
generated_artifacts: []
tags:
  - agentqms
  - context
  - architecture
  - fragmentation
  - assessment
---

# Context System Architecture: Fragmentation Analysis

## Executive Summary

**Finding**: You have **THREE SEPARATE CONTEXT SYSTEMS** from different architectural iterations, causing confusion and preventing successful implementation.

**Systems Identified**:
1. **Modern Plugin-Based System** (Current, 2026-01-05+) ✅
2. **Legacy Copilot System** (Archived, pre-2026) ❌
3. **Utility Helper Scripts** (Transitional, mixed) ⚠️

**Recommendation**: **Consolidate on System #1** (Plugin-Based), deprecate System #2, integrate useful parts of System #3.

---

## Part 1: The Three Context Systems

### **System #1: Modern Plugin-Based Context Bundles** ✅

**Status**: Current, Active, Best Architecture
**Location**: `AgentQMS/.agentqms/plugins/context_bundles/`

**Core Files**:
```
AgentQMS/tools/core/context_bundle.py          # Main implementation
AgentQMS/.agentqms/plugins/context_bundles/    # Plugin definitions
  └── security-review.yaml                      # Example bundle
```

**How It Works**:
```python
# Direct usage
from AgentQMS.tools.core.context_bundle import get_context_bundle

# Automatic task detection
files = get_context_bundle("implement new feature")

# Explicit bundle
files = get_context_bundle("security review", task_type="security-review")
```

**Architecture**:
- Integrates with **plugin system** (can discover community bundles)
- Supports **tiered context** (tier1, tier2, tier3)
- Has **freshness checking** (ensures docs are recent)
- Uses **glob patterns** for file matching
- Supports **task type classification** (development, debugging, planning, docs)

**Definition Example**:
```yaml
# AgentQMS/.agentqms/plugins/context_bundles/security-review.yaml
name: security-review
version: '1.0'
description: Security review context bundle
tiers:
  tier1:
    files:
      - AgentQMS/standards/tier1-sst/security-standards.md
      - docs/guides/security-checklist.md
  tier2:
    files:
      - AgentQMS/standards/tier2-framework/security-framework.yaml
```

**Strengths**:
- ✅ Plugin-based (extensible)
- ✅ Modular and composable
- ✅ Integrated with validation system
- ✅ Freshness guarantees
- ✅ Task classification

**Current Usage**: Imported by `validate_artifacts.py`, available via plugin registry

---

### **System #2: Legacy Copilot Workflow System** ❌

**Status**: Archived, Deprecated
**Location**: `archive/archive_agentqms/DEPRECATED/legacy-agent-configs/.copilot/context/`

**Core Files**:
```
.copilot/context/workflow-triggers.yaml         # ARCHIVED (found in archive/)
```

**How It Worked**:
- Used by GitHub Copilot / VS Code extension
- Workflow triggers mapped tasks to file patterns
- Probably worked with `suggest_context.py` utility

**Why It's Archived**:
- Old VS Code/Copilot-specific approach
- Not portable across agents
- Replaced by plugin system

**Status**: **DEPRECATED** - Do not use

---

### **System #3: Utility Helper Scripts** ⚠️

**Status**: Transitional, Mixed (Some useful, some outdated)
**Location**: `AgentQMS/tools/utilities/`

#### **Files Analysis**:

##### **A. `get_context.py`** ⚠️ **Transitional**
```python
# Canonical implementation - bridges to plugin system
from AgentQMS.tools.core.context_bundle import get_context_bundle

# Has fallback to deprecated handbook index
```

**Status**: **KEEP** - Useful CLI wrapper
**Purpose**: Command-line interface to context bundles
**Usage**:
```bash
python AgentQMS/tools/utilities/get_context.py --task "implement feature"
python AgentQMS/tools/utilities/get_context.py --list-context-bundles
```

**Assessment**: This is a **useful adapter** that provides CLI access to the plugin system.

---

##### **B. `suggest_context.py`** ⚠️ **CONFLICTING**
```python
# Loads from .copilot/context/workflow-triggers.yaml
self.triggers_file = self.project_root / ".copilot" / "context" / "workflow-triggers.yaml"
```

**Status**: **BROKEN** - References archived file
**Problem**: Tries to load `workflow-triggers.yaml` which is now archived
**Purpose**: Keyword-based context suggestion

**Assessment**: **NEEDS UPDATE** - Should use plugin system instead of archived triggers

**Fix Needed**:
```python
# CURRENT (BROKEN):
triggers_file = ".copilot/context/workflow-triggers.yaml"  # ARCHIVED!

# SHOULD BE:
from AgentQMS.tools.core.plugins import get_plugin_registry
registry = get_plugin_registry()
bundles = registry.get_context_bundles()
```

---

##### **C. `smart_populate.py`** ✅ **INDEPENDENT**
**Status**: **KEEP** - Useful utility
**Purpose**: Auto-populate artifact fields from git context
**Not Part of Context System**: This is for artifact creation, not context loading

**Assessment**: **UNRELATED** - Useful but separate concern

---

##### **D. `agent_feedback.py`** ✅ **INDEPENDENT**
**Status**: **KEEP** - Useful utility
**Purpose**: Collect agent feedback about documentation issues
**Not Part of Context System**: Feedback collection, not context loading

**Assessment**: **UNRELATED** - Useful but separate concern

---

##### **E. `versioning.py`** ❓ **UNKNOWN**
**Not examined** - Likely unrelated to context system

---

### **System #4: YAML Configuration Files** 🤔 **UNCLEAR PURPOSE**

**Files**:
```
AgentQMS/standards/context_classification.yaml
AgentQMS/standards/context_map.yaml
```

#### **`context_classification.yaml`**
**Purpose**: Task classification rules (keywords → context type)
```yaml
classification_rules:
  - pattern: "experiment"
    keywords: ["experiment", "etk", "tracker"]
    context_type: "ocr_experiment"
    confidence_threshold: 0.7
```

**Status**: **PARTIALLY REDUNDANT**
**Why**: `context_bundle.py` already has `TASK_KEYWORDS` for classification
**Assessment**: Could be **consolidated** - either:
1. Remove this file and use `context_bundle.py` classification, OR
2. Load this YAML in `context_bundle.py` instead of hardcoding

---

#### **`context_map.yaml`**
**Purpose**: Maps task types to file patterns
```yaml
task_mappings:
  ocr_experiment:
    core_files: ["experiment_manager/etk.py", ...]
    config_files: ["configs/**/*.yaml", ...]
  documentation_update:
    doc_files: ["docs/**/*.md", ...]
```

**Status**: **OVERLAPS WITH PLUGIN BUNDLES**
**Why**: Plugin context bundles already define file mappings
**Assessment**: This looks like an **early prototype** of what became context bundles

**Comparison**:
```yaml
# context_map.yaml (FLAT)
task_mappings:
  ocr_experiment:
    core_files: [...]
    config_files: [...]

# context_bundle plugin (TIERED)
name: ocr-experiment
tiers:
  tier1:
    files: [...]  # Essential
  tier2:
    files: [...]  # Detailed
```

**Recommendation**: **MIGRATE** to plugin bundles OR **DEPRECATE**

---

## Part 2: Why Implementation Failed

### **Root Causes**:

1. **Multiple Entry Points**
   - `get_context.py` → Wrapper (calls `context_bundle.py`)
   - `suggest_context.py` → Broken (references archived file)
   - `context_bundle.py` → Correct (plugin-based)
   - **Problem**: Which one to use? Not clear!

2. **Missing Connection**
   - **YAML configs exist** (`context_classification.yaml`, `context_map.yaml`)
   - **Plugin bundles exist** (`security-review.yaml`)
   - **Core loader exists** (`context_bundle.py`)
   - **Problem**: Not wired together!

3. **No Documentation**
   - Which system is current?
   - How to create bundles?
   - How to use from agents?
   - **Problem**: No clear usage guide!

4. **Partial Migration**
   - Old system archived but utilities still reference it
   - YAML configs don't match plugin structure
   - **Problem**: Incomplete transition!

---

## Part 3: Consolidation Plan

### **Phase 1: Fix Immediate Breakage** 🔥

#### **1.1 Fix `suggest_context.py`**

**Current (BROKEN)**:
```python
# Tries to load archived file
self.triggers_file = ".copilot/context/workflow-triggers.yaml"
```

**Fix**:
```python
# Use plugin registry instead
from AgentQMS.tools.core.plugins import get_plugin_registry

def _load_triggers(self):
    """Load triggers from plugin context bundles."""
    registry = get_plugin_registry()
    bundles = registry.get_context_bundles()

    # Build task_types from bundle definitions
    for name, bundle in bundles.items():
        # Extract keywords from bundle description or metadata
        ...
```

#### **1.2 Consolidate Classification**

**Option A**: Use YAML config
```python
# In context_bundle.py
def load_classification_rules():
    """Load from context_classification.yaml"""
    path = PROJECT_ROOT / "AgentQMS/standards/context_classification.yaml"
    with open(path) as f:
        return yaml.safe_load(f)
```

**Option B**: Deprecate YAML, keep Python dict
```python
# Keep TASK_KEYWORDS in context_bundle.py
# Delete context_classification.yaml
```

**Recommendation**: **Option B** (simpler, one source of truth)

#### **1.3 Migrate or Deprecate `context_map.yaml`**

**Option A**: Create plugin bundles from it
```bash
# Convert context_map.yaml → plugin YAML files
python convert_context_map_to_plugins.py
```

**Option B**: Delete it (if already covered by plugins)
```bash
# Check if plugins exist for all mappings
# If yes, delete context_map.yaml
```

**Recommendation**: **Option A** if mappings are still useful

---

### **Phase 2: Establish Clear Entry Points** 📍

#### **2.1 Primary API**
```python
# AgentQMS/tools/core/context_bundle.py
def get_context_bundle(description: str, task_type: str | None = None) -> list[Path]:
    """
    Get context files for a task.

    This is the ONE CANONICAL way to get context.
    """
```

#### **2.2 CLI Wrapper**
```bash
# AgentQMS/tools/utilities/get_context.py
python get_context.py --task "implement feature"  # Uses context_bundle.py
```

#### **2.3 Deprecate Redundant Tools**
```bash
# suggest_context.py → Fix or merge into get_context.py
# context_classification.yaml → Delete or integrate
# context_map.yaml → Migrate to plugins or delete
```

---

### **Phase 3: Integration with Agent Workflows** 🤖

This connects to the **Plugin Evolution Roadmap** from the earlier research doc.

#### **3.1 Automatic Context Injection**
```yaml
# In context bundle definitions
name: security-review
triggers: [security, review, audit, compliance]  # NEW
auto_inject: true  # NEW
priority: high  # NEW
```

#### **3.2 MCP Server Integration**
```python
# In MCP server
@server.tool()
async def get_task_context(task: str):
    """Automatically inject relevant context."""
    files = get_context_bundle(task)
    return assemble_context(files)
```

#### **3.3 Agent Configuration**
```yaml
# AGENTS.yaml
agents:
  security_reviewer:
    context_bundles:
      - security-review  # Auto-loaded
    triggers:
      - security
      - vulnerability
```

---

## Part 4: Implementation Checklist

### **Immediate (Week 1)**
- [ ] Fix `suggest_context.py` to use plugin registry
- [ ] Decide on `context_classification.yaml` (delete or integrate)
- [ ] Audit `context_map.yaml` (migrate to plugins or deprecate)
- [ ] Document canonical usage pattern

### **Short Term (Week 2-4)**
- [ ] Create missing plugin bundles for common tasks
- [ ] Write usage guide: "How to Create Context Bundles"
- [ ] Add examples to plugin system research doc
- [ ] Test with real agents (Claude, Copilot, etc.)

### **Medium Term (Month 2-3)**
- [ ] Implement Phase 1 from Plugin Evolution (auto-injection)
- [ ] Integrate with MCP server
- [ ] Add context selection to agent interface
- [ ] Build context selector UI/CLI

---

## Part 5: File-by-File Recommendations

| File | Status | Action | Priority |
|------|--------|--------|----------|
| **`context_bundle.py`** | ✅ Current | **KEEP** - This is the core | P0 |
| **`get_context.py`** | ⚠️ Wrapper | **KEEP** - Useful CLI | P1 |
| **`suggest_context.py`** | ❌ Broken | **FIX** - Update to use plugins | P0 |
| **`smart_populate.py`** | ✅ Independent | **KEEP** - Unrelated utility | P2 |
| **`agent_feedback.py`** | ✅ Independent | **KEEP** - Unrelated utility | P2 |
| **`context_classification.yaml`** | ⚠️ Redundant | **DELETE** or integrate | P1 |
| **`context_map.yaml`** | ⚠️ Overlapping | **MIGRATE** to plugins or delete | P1 |
| **`.copilot/workflow-triggers.yaml`** | ❌ Archived | **ALREADY DELETED** (archived) | Done |

---

## Part 6: Architecture Decision

### **Chosen System**: Plugin-Based Context Bundles (System #1)

**Rationale**:
1. **Most Modern**: Written with 2026-01-05 timestamp
2. **Best Architecture**: Pluggable, extensible, composable
3. **Most Complete**: Has all features (tiers, freshness, classification)
4. **Integrated**: Works with plugin registry and validation
5. **Future-Proof**: Aligns with Plugin Evolution roadmap

### **Migration Path**:
```
Legacy Systems (2, 3, 4)
    ↓
  FIX broken references
    ↓
  MIGRATE useful patterns to plugins
    ↓
  DEPRECATE old files
    ↓
Modern Plugin System (1)
    ↓
  ENHANCE with auto-injection (Phase 1)
    ↓
  INTEGRATE with agents (Phase 2)
```

---

## Part 7: Quick Start Guide (After Consolidation)

### **For Agent Developers**:

```python
# 1. Get context automatically
from AgentQMS.tools.core.context_bundle import get_context_bundle

files = get_context_bundle("implement security feature")
# Returns: [Path(...), Path(...), ...]

# 2. Load and assemble context
context = assemble_context_from_files(files)

# 3. Inject into prompt
prompt = f"{context}\n\nTask: {user_request}"
```

### **For Creating New Bundles**:

```yaml
# AgentQMS/.agentqms/plugins/context_bundles/my-bundle.yaml
name: my-bundle
version: '1.0'
description: My custom context bundle

tiers:
  tier1:
    description: Essential context
    files:
      - path/to/essential/*.md

  tier2:
    description: Detailed context
    files:
      - path/to/detailed/*.yaml
```

### **For Command Line**:

```bash
# List available bundles
python AgentQMS/tools/utilities/get_context.py --list-context-bundles

# Get context for task
python AgentQMS/tools/utilities/get_context.py --task "debug memory leak"

# Get specific bundle
python AgentQMS/tools/utilities/get_context.py --bundle security-review
```

---

## Conclusion

**Problem**: Three separate context systems from different architectural attempts caused confusion and implementation failure.

**Solution**: Consolidate on **Plugin-Based Context Bundles** (System #1), fix broken utilities, deprecate outdated configs.

**Timeline**:
- **Week 1**: Fix breakage, make decisions
- **Week 2-4**: Create bundles, document usage
- **Month 2+**: Enhance with auto-injection

**Success Criteria**:
- ✅ One clear entry point (`context_bundle.py`)
- ✅ All utilities working and referencing correct system
- ✅ Documentation showing how to use
- ✅ Examples of bundles created and working
- ✅ Integration with agents demonstrated

---

## Next Steps

1. **Decide**: Delete or integrate `context_classification.yaml`?
2. **Audit**: What's in `context_map.yaml` that's not in plugins?
3. **Fix**: Update `suggest_context.py` to use plugin registry
4. **Document**: Write "Context Bundles User Guide"
5. **Test**: Try using bundles with real agent workflows
