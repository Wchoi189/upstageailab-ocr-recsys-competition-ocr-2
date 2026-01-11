# Utility Scripts Discovery & Context Bundling — Visual Summary

## Problem-Solution Map

```
PROBLEM                          SOLUTION
─────────────────────────────────────────────────────────────

Agent writes custom YAML loading  ConfigLoader (caching, fallbacks)
Agent hardcodes paths             paths.get_project_root(), etc.
Agent calls subprocess git        git.get_current_branch()
Agent creates custom timestamp    timestamps.get_kst_timestamp()
Agent doesn't know utilities exist  ← DISCOVERY SYSTEM (THIS)
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  UTILITY SCRIPTS IN CODEBASE                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  config_loader.py     git.py         paths.py              │
│  ├─ ConfigLoader      ├─ get_branch   ├─ get_project_root  │
│  ├─ load_yaml()       ├─ get_commit   ├─ get_artifacts_dir │
│  └─ load_config()     └─ validate()   └─ get_docs_dir()    │
│                                                              │
│  timestamps.py        runtime.py     sync_github_projects  │
│  ├─ get_kst_ts()      └─ ensure_path ├─ GitHubManager      │
│  ├─ parse_ts()                       └─ create_issue()     │
│  └─ get_age()                                              │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ (catalog via)
                            │
┌─────────────────────────────────────────────────────────────┐
│  DISCOVERY LAYER (TO BE BUILT)                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  context/utility-scripts/                                   │
│  ├─ UTILITY_SCRIPTS_INDEX.yaml      (searchable index)     │
│  ├─ QUICK_REFERENCE.md              (one-pager)           │
│  ├─ by-category/                    (organized docs)       │
│  │  ├─ configuration/               │                      │
│  │  ├─ path-resolution/             │                      │
│  │  ├─ timestamps/                  │                      │
│  │  ├─ git/                         │                      │
│  │  └─ github/                      │                      │
│  └─ by-use-case/                    (problem → solution)    │
│     ├─ "I need to load config"      │                      │
│     ├─ "I need to find a path"      │                      │
│     └─ ...                          │                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ (accessed via)
                            │
┌─────────────────────────────────────────────────────────────┐
│  AI AGENT INTEGRATION                                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Option A: Auto Context Bundling                           │
│  ├─ Agent asks about "config loading"                      │
│  └─ Context system auto-includes utility-scripts bundle    │
│                                                              │
│  Option B: MCP Tool                                        │
│  ├─ Agent calls: list_utilities(category="configuration")  │
│  └─ Returns: ConfigLoader with examples                    │
│                                                              │
│  Option C: Prompt Guidance                                 │
│  ├─ Agent instructions include QUICK_REFERENCE.md          │
│  └─ Agent checks table before writing custom code          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Discovery Workflow Example

```
Agent Task: "Create an artifact and load its metadata from YAML config"

┌─────────────────────────────────────┐
│ 1. Agent Plans Implementation        │
│    - Need to load YAML config       │
│    - Need to find artifact location │
│    - Need to get current timestamp  │
└──────────────┬──────────────────────┘
               │ (consults context or tool)
               ▼
┌─────────────────────────────────────┐
│ 2. Discovery System Responds         │
│    "For YAML config loading:"        │
│    → ConfigLoader.load_yaml()        │
│                                      │
│    "For artifact location:"          │
│    → paths.get_artifacts_dir()       │
│                                      │
│    "For timestamp:"                  │
│    → timestamps.get_kst_timestamp()  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. Agent Uses Discovered Utilities   │
│    from config_loader import Config │
│    from paths import get_artifacts  │
│    from timestamps import get_kst   │
│                                      │
│    config = ConfigLoader.load_yaml() │
│    artifact_dir = get_artifacts_dir()│
│    ts = get_kst_timestamp()          │
└─────────────────────────────────────┘
   ✓ No reinvention
   ✓ Consistent APIs
   ✓ Performance optimized (caching)
```

## Implementation Timeline

```
Phase 1: Documentation          (2-4h) → Immediate value
├─ Markdown files per utility
├─ Quick reference summary
└─ YAML index

Phase 2: Context Integration    (2-3h) → Auto discovery
├─ Bundle into context system
├─ Keyword-based triggering
└─ Test with sample queries

Phase 3: MCP Tool [Optional]     (3-4h) → Programmatic access
├─ list_utilities() tool
├─ Search/filter logic
└─ Agent integration

Phase 4: Auto-Generation [Future](4-6h) → Self-maintaining
├─ Scan source code at startup
├─ Extract docstrings/examples
├─ Auto-rebuild index
└─ Watch for changes
```

## Key Decisions

### 1. Discovery Method: Which Approach?

| Approach | Pros | Cons | Effort |
|----------|------|------|--------|
| **Auto Context** | Seamless, automatic | Requires context system | Medium |
| **MCP Tool** | Explicit, agent-controlled | Requires tool call overhead | Medium |
| **Prompt Guidance** | Simple, immediate | Manual, easily forgotten | Low |
| **Auto-Generation** | Always in sync | Complex, maintenance needed | High |

**Recommendation**: Start with **Prompt Guidance + Phase 1 Docs**, then add **Phase 2 Context** when context system supports it.

### 2. Scope: Which Utilities First?

**High Priority** (Highest reuse):
- `config_loader.py` — Used by artifact system, MCP servers
- `paths.py` — Used everywhere paths are resolved
- `timestamps.py` — Required for artifact metadata

**Medium Priority**:
- `git.py` — Used for branch metadata
- `config.py` — Higher-level config merging

**Lower Priority**:
- `runtime.py` — Very specific (path setup)
- `sync_github_projects.py` — Specialized (GitHub only)

**Recommendation**: Document all, but highlight top 3 in QUICK_REFERENCE.

### 3. Maintenance: Keep It Updated

When new utilities added:
1. Add entry to UTILITY_SCRIPTS_INDEX.yaml
2. Create markdown doc in by-category/
3. Add to QUICK_REFERENCE.md if high-value

**Responsibility**: Developer who adds utility documents it.

---

## Expected Outcomes

### Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Discoverability** | Agent finds utility in <10 sec | Time from "need X" to "found Y" |
| **Adoption** | 70% of new code uses discovered utils | Code review analysis |
| **Duplication** | 80% reduction in custom implementations | Grep for reimplemented functions |
| **Consistency** | All code uses same patterns | API uniformity across project |
| **Performance** | Measurable caching benefits | ConfigLoader cache hit rates |

### Example: Before vs. After

**Before** (Without Discovery):
```python
# Agent writes custom YAML loading
import yaml
with open("config.yaml") as f:
    config = yaml.safe_load(f)
if not config:
    config = {"timeout": 30}
# No caching, no type safety, duplicated everywhere
```

**After** (With Discovery):
```python
# Agent uses ConfigLoader
from AgentQMS.tools.utils.config_loader import ConfigLoader
config = ConfigLoader.load_yaml("config.yaml",
                                defaults={"timeout": 30})
# Cached, consistent, fallback-safe, reusable
```

---

## Next Steps

1. **Decide**: Which approach appeals most?
   - Start with docs + context? (Recommended)
   - Go straight to MCP tool?
   - Hybrid?

2. **Timeline**: Phase 1 can be done in one session (2-4h)

3. **Iteration**: Get feedback on Phase 1 before moving to Phase 2

4. **Ownership**: Who maintains utility documentation?

---

## Questions for Discussion

1. **Automation Level**: Manual docs vs. auto-generated + hand-written?
2. **Discoverability**: Context bundling vs. MCP tool vs. prompt?
3. **Scope**: Focus on top 3 utilities first, or document all?
4. **Maintenance**: How to keep documentation in sync with code?
5. **Testing**: How to verify agents actually use utilities?

---

## Reference

Full requirements document: [UTILITY_SCRIPTS_DISCOVERY_REQUIREMENTS.md](./UTILITY_SCRIPTS_DISCOVERY_REQUIREMENTS.md)

Utility catalog (generated): [utility_catalog.txt](/tmp/utility_catalog.txt)
