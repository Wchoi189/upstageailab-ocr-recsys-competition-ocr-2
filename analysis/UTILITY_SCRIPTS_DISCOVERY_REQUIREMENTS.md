# Utility Scripts Discovery & Context Bundling — Requirements Brainstorm

**Date**: 2026-01-11
**Status**: Requirements Analysis (Pre-Implementation)
**Goal**: Design a system to help AI agents discover and reuse existing utility scripts instead of inventing new wheels

---

## 1. Problem Statement

### Current State
- **AgentQMS/tools/utils/** contains 7 reusable utility modules with ~50+ public functions
- Examples: `config_loader.py`, `git.py`, `paths.py`, `timestamps.py`, `runtime.py`, `sync_github_projects.py`
- **Issue**: AI agents often don't know these utilities exist, so they:
  - Write custom code to do things already solved
  - Duplicate logic across the codebase
  - Miss performance optimizations (e.g., caching in ConfigLoader)
  - Create inconsistent APIs (multiple config loading patterns)

### Why This Matters
- **Code duplication** increases maintenance burden
- **Inconsistent patterns** make codebase harder to understand
- **Performance loss** from reimplemented utilities without optimization
- **DevEx degradation** as the codebase grows

---

## 2. What Needs to Be Discovered?

### 2.1 Utility Types to Catalog

```
config_loader.py
├── ConfigLoader class (stateful, with caching)
├── load_yaml() (static, simple)
├── load_config() (module-level convenience)
└── _extract_nested() (helper for dot-notation)

git.py
├── get_current_branch() — detect current git branch
├── get_default_branch() — config-based default
├── validate_branch_name() — name validation
├── get_default_branch_from_remote() — remote detection
├── is_in_git_repository() — repo check
└── get_git_commit_for_file() — file history lookup

paths.py
├── get_framework_root() — locate AgentQMS
├── get_project_root() — locate project root
├── get_container_path() — config-based path resolution
├── get_artifacts_dir() — standard artifact location
├── get_docs_dir() — standard docs location
├── get_agent_interface_dir() — agent interface location
├── get_agent_tools_dir() — tools location
├── get_project_conventions_dir() — conventions location
├── get_project_config_dir() — config location
└── ensure_within_project() — safety check

timestamps.py
├── get_configured_timezone() — environment-aware timezone
├── get_timezone_abbr() — IANA to abbreviation
├── get_kst_timestamp() — KST-formatted timestamp
├── parse_timestamp() — parse timestamp string
├── get_age_in_days() — age calculation
├── format_timestamp_for_filename() — filename-safe timestamp
└── infer_artifact_filename_timestamp() — extract from filename

runtime.py
└── ensure_project_root_on_sys_path() — sys.path setup

sync_github_projects.py
└── GitHubManager class (GitHub Project V2 integration)
```

### 2.2 Discovery Dimensions

What questions should an AI agent be able to answer?

| Question | Example | Benefit |
|----------|---------|---------|
| **"What can I use for X?"** | "I need to load YAML config" → ConfigLoader | Avoids reimplementation |
| **"What functions exist?"** | "List all path utilities" | Reduces trial-and-error |
| **"How do I use this?"** | "Show me config_loader examples" | Speeds up adoption |
| **"Are there any gotchas?"** | "ConfigLoader uses caching — watch out for stale data" | Prevents bugs |
| **"What does this depend on?"** | "ConfigLoader needs PyYAML" | Prevents import errors |
| **"What's the performance impact?"** | "ConfigLoader caches; ~2000x speedup on repeats" | Informs design decisions |

---

## 3. Context Bundling Strategy

### 3.1 What Should Be Bundled?

For each utility module, capture:

```yaml
module: config_loader.py
summary: "Framework-agnostic YAML loading with caching"
purpose: "Centralized configuration loading with graceful fallbacks"
location: "AgentQMS/tools/utils/"
dependencies:
  - PyYAML (optional; graceful fallback if missing)
  - pathlib (stdlib)
exports:
  classes:
    - name: ConfigLoader
      purpose: "Stateful loader with LRU caching"
      key_methods:
        - load_yaml() → static, simple one-off loads
        - get_config() → instance, cached, supports nested keys
        - clear_cache() → debug/reset
      caching: "LRU with configurable size (default 10)"
      performance: "~2000x speedup for cached loads"

  functions:
    - name: load_config()
      purpose: "Convenience function using module-level loader"
      signature: "load_config(path, key=None) → dict | Any"
      parameters:
        - path: "Path to YAML file"
        - key: "Optional nested key (dot notation: 'a.b.c')"
      returns: "Full config dict or nested value"
      fallback: "Empty dict if file not found or YAML unavailable"

examples:
  - name: "Simple YAML load"
    code: |
      config = ConfigLoader.load_yaml("config.yaml")
  - name: "With defaults"
    code: |
      config = ConfigLoader.load_yaml("config.yaml",
                                      defaults={"timeout": 30})
  - name: "Cached instance with nested key"
    code: |
      loader = ConfigLoader()
      port = loader.get_config("config.yaml", key="server.port")

usage_patterns:
  - "Use ConfigLoader for repeated access (caching)"
  - "Use load_yaml() for one-off loads"
  - "Prefer instances when loading multiple configs"

error_handling:
  - "Gracefully returns defaults if file not found"
  - "Works without PyYAML (returns empty dict)"
  - "Does not raise exceptions on YAML parse errors"

when_to_use:
  - "Loading framework or project config"
  - "MCP servers need to load tool/resource configs"
  - "Any multi-file configuration with caching benefits"

when_not_to_use:
  - "One-time config loads in startup (marginal benefit)"
  - "Configuration that changes at runtime (caching is stale)"
  - "Sensitive configs that need real-time file checking"
```

### 3.2 Bundle Structure

```
context/utility-scripts/
├── UTILITY_SCRIPTS_INDEX.yaml        # Searchable index
├── by-category/
│   ├── configuration/
│   │   └── config_loader.md
│   ├── path-resolution/
│   │   └── paths.md
│   ├── timestamps/
│   │   └── timestamps.md
│   ├── git/
│   │   └── git.md
│   └── github/
│       └── sync_github_projects.md
├── by-use-case/
│   ├── "I need to load config"
│   │   └── → config_loader.md
│   ├── "I need to find a path"
│   │   └── → paths.md
│   └── ...
└── QUICK_REFERENCE.md               # TL;DR for busy agents
```

---

## 4. Discovery Mechanism

### 4.1 How Will AI Agents Find This?

**Option A: Explicit Bundling (Recommended)**
- Create `context/utility-scripts/` with markdown files
- Bundle gets loaded when context is requested
- Indexed by category, use-case, function name
- Updated manually when new utilities added

**Option B: Auto-generated Catalog**
- Scan `AgentQMS/tools/utils/` at startup
- Extract docstrings, signatures, examples
- Generate markdown or JSON catalog
- Rebuild on demand or on filesystem watch

**Option C: Hybrid**
- Auto-scan for raw facts (functions, signatures)
- Hand-written markdown for guidance, examples, patterns
- Combined into searchable index

**Option D: MCP Tool (Advanced)**
- Create `list_utilities()` tool that agents can call
- Returns filtered list by category, purpose, dependencies
- Can answer "what can I use for X?" queries
- Requires tool implementation

### 4.2 Search Interface

Agents should be able to query:

```python
# By purpose
"I need to load YAML config"
→ config_loader.ConfigLoader, config_loader.load_yaml()

# By function name
"find functions for getting paths"
→ paths.get_project_root, paths.get_framework_root, ...

# By dependency
"what utils work without external deps?"
→ runtime.ensure_project_root_on_sys_path, ...

# By category
"show all git utilities"
→ [get_current_branch, get_default_branch, ...]

# By keyword
"caching" → config_loader (LRU cache, performance tips)
```

---

## 5. Context Bundle Contents

### 5.1 Per-Utility Document Structure

```markdown
# ConfigLoader — YAML Configuration Loading

## Quick Summary
Framework-agnostic configuration loader with LRU caching and graceful fallbacks.

## Purpose
Centralized, cached YAML loading for configuration files across the project.

## Location
`AgentQMS/tools/utils/config_loader.py`

## When to Use
- Loading framework or project configuration
- Repeated access to same config (caching benefit)
- MCP servers loading tool/resource definitions
- Any scenario where you need cached, nested key extraction

## When NOT to Use
- Configuration that changes at runtime (stale cache issue)
- One-time loads without repeated access
- Sensitive configs requiring real-time file checking

## Key APIs

### ConfigLoader.load_yaml() — Static Method
Load YAML with fallback defaults (no caching).

```python
config = ConfigLoader.load_yaml("config.yaml",
                                defaults={"timeout": 30})
```

**Use when**: One-off loads, simple cases.

### ConfigLoader.get_config() — Instance Method
Load YAML with LRU caching and optional nested key extraction.

```python
loader = ConfigLoader(cache_size=5)
port = loader.get_config("config.yaml", key="server.port",
                         defaults=8080)
```

**Use when**: Repeated loads, nested keys, performance matters.

## Examples

[Practical examples with context and expected output]

## Performance Characteristics
- First load: ~8ms (disk I/O)
- Cached load: ~0.004ms (2000x faster)
- Cache overhead: Minimal (dict in memory)

## Dependencies
- Optional: PyYAML (graceful fallback if missing)
- No external dependencies otherwise

## Error Handling
- File not found → Returns defaults (no exception)
- YAML parse error → Returns defaults (no exception)
- PyYAML unavailable → Returns defaults (no exception)

## Integration Examples
- `artifact_templates.py` uses ConfigLoader for template defaults
- `unified_server.py` uses ConfigLoader for tool/resource configs
- Can be adopted wherever inline YAML loading exists

## Gotchas & Notes
- **Caching issue**: If config file changes at runtime, cache won't see updates. Call `loader.clear_cache()` to force reload.
- **Type checking**: Returned values from nested keys are `Any`. Validate/cast as needed.
- **Thread safety**: Not thread-safe; use single instance per thread or lock access.

## Related Utilities
- `config.py` — Higher-level config merging (framework + project + env)
- `paths.py` — Path resolution (often used with loaded config)
```

### 5.2 Index Document

```yaml
utilities:
  - name: config_loader
    file: config_loader.py
    summary: "YAML configuration loading with caching"
    categories: [configuration, caching, yaml]
    use_cases: ["Load configuration", "Cache repeated access", "Extract nested values"]
    key_exports: [ConfigLoader, load_yaml, load_config]

  - name: paths
    file: paths.py
    summary: "Consistent path resolution for project components"
    categories: [paths, filesystem, configuration]
    use_cases: ["Find project root", "Locate artifacts", "Get standard directories"]
    key_exports: [get_project_root, get_artifacts_dir, get_docs_dir, ...]

  # ... more utilities
```

---

## 6. AI Agent Integration Points

### 6.1 When Should Agents Use This?

**Scenario 1: Writing Code That Uses Paths**
- Agent needs to find where artifacts are stored
- Instead of hardcoding `"docs/artifacts/"`
- Query: "What utility finds artifact directories?"
- Result: `paths.get_artifacts_dir()`

**Scenario 2: Loading Configuration**
- Agent needs to load a YAML config file
- Instead of writing inline `yaml.safe_load()` with try/except
- Query: "How do I load a YAML config with fallback?"
- Result: `ConfigLoader.load_yaml(path, defaults=...)`

**Scenario 3: Git Integration**
- Agent needs to detect current branch for artifact metadata
- Instead of calling `subprocess.run(['git', ...])`
- Query: "What utility gets current git branch?"
- Result: `git.get_current_branch()`

### 6.2 How Is This Made Available?

**Option 1: Context Bundling (Automatic)**
- When asking about file paths, relevant context auto-loads
- Search engine ranks utility-scripts high for relevant queries
- Agent sees "Consider using paths.py utilities:" in response

**Option 2: MCP Tool (Explicit)**
```python
@app.list_tools()
async def list_utilities():
    """List available utility scripts"""
    return [
        Tool(
            name="list_utilities",
            description="Search reusable utility scripts by purpose/category",
            inputSchema={...}
        )
    ]

# Agent calls:
# list_utilities(query="I need to load YAML config")
# → Returns ConfigLoader with examples
```

**Option 3: README/Docs (Manual)**
- Document in `AgentQMS/README.md`: "Available Utilities"
- Link to utility index
- Agent reads during onboarding

### 6.3 Prompt Guidance

Include in agent instructions:

```markdown
## Reusable Utilities

Before writing code to solve these problems, check if a utility exists:

| Problem | Utility | Module |
|---------|---------|--------|
| Load YAML config | ConfigLoader | config_loader.py |
| Get project root | get_project_root() | paths.py |
| Get artifacts dir | get_artifacts_dir() | paths.py |
| Get current branch | get_current_branch() | git.py |
| Get timestamp | get_kst_timestamp() | timestamps.py |

**Reference**: `context/utility-scripts/QUICK_REFERENCE.md`
```

---

## 7. Implementation Phases

### Phase 1: Documentation (Immediate)
- [ ] Create `context/utility-scripts/` directory structure
- [ ] Write markdown files for each utility
- [ ] Create QUICK_REFERENCE.md summary
- [ ] Create searchable YAML index
- **Effort**: 2-4 hours
- **Benefit**: Immediate, manual discoverability

### Phase 2: Context Integration (Next)
- [ ] Bundle utility-scripts into context detection system
- [ ] Trigger bundle when relevant keywords detected
- [ ] Test with sample agent queries
- **Effort**: 2-3 hours
- **Benefit**: Automatic injection into relevant conversations

### Phase 3: MCP Tool (Optional)
- [ ] Create `list_utilities` tool in unified_server.py
- [ ] Implement search/filter logic
- [ ] Test with agent queries
- **Effort**: 3-4 hours
- **Benefit**: Agents can query programmatically

### Phase 4: Auto-Generation (Future)
- [ ] Scan utils/ directory at startup
- [ ] Extract docstrings, signatures, examples
- [ ] Generate index automatically
- [ ] Watch for changes, rebuild index
- **Effort**: 4-6 hours
- **Benefit**: Self-maintaining, always in sync

---

## 8. Success Criteria

### How Will We Know This Works?

1. **Discoverability**: Can an AI agent find relevant utility in <10 seconds?
2. **Usage**: Does new code prefer existing utils over reimplementation?
3. **Consistency**: Are multiple codebases using same patterns (e.g., ConfigLoader)?
4. **Performance**: Do agents avoid writing slow solutions (e.g., custom config caching)?
5. **Maintenance**: Is there single source of truth for utility APIs?

### Metrics to Track
- % of new code that uses discovered utilities
- Number of duplicate implementations reduced
- Agent queries that result in utility recommendation
- Cache hit rates in production (ConfigLoader performance)

---

## 9. Open Questions for Discussion

1. **Priority**: Should discovery be automatic (bundle) or explicit (MCP tool)?
   - Trade-off: Ease of use vs. Agent autonomy

2. **Scope**: Which utilities to catalog first?
   - Start with: config_loader, paths, git (highest reuse potential)
   - Defer: sync_github_projects (more specialized)

3. **Updates**: How often should utility documentation be refreshed?
   - Manual on-demand? Monthly? Automated?

4. **Training**: Should agent instructions explicitly list utilities?
   - Risk of information overload vs. discoverability

5. **Testing**: How to verify agents actually use utilities?
   - Monitor generated code? Track import statements?

---

## 10. Recommended Next Steps

1. **Decide on Bundling Strategy**: Auto vs. Manual vs. Hybrid
2. **Start with Phase 1**: Create documentation bundle
3. **Pilot with ConfigLoader**: One utility as proof-of-concept
4. **Gather Feedback**: Test with actual agent workflows
5. **Iterate**: Refine based on real-world usage

---

## Appendix: Existing Utilities Quick Reference

| Utility | Purpose | Key Functions | When to Use |
|---------|---------|---------------|------------|
| **config_loader.py** | YAML config loading with caching | `ConfigLoader`, `load_yaml()` | Loading configs, MCP servers |
| **paths.py** | Consistent path resolution | `get_project_root()`, `get_artifacts_dir()` | Finding standard directories |
| **timestamps.py** | Timezone-aware timestamp handling | `get_kst_timestamp()`, `get_age_in_days()` | Artifact metadata, timestamps |
| **git.py** | Git branch/commit detection | `get_current_branch()`, `validate_branch_name()` | Artifact metadata, git info |
| **runtime.py** | Runtime path setup | `ensure_project_root_on_sys_path()` | Module initialization |
| **config.py** | Hierarchical config merging | `ConfigLoader`, `load_config()` | Framework config (higher-level than config_loader.py) |
| **sync_github_projects.py** | GitHub Project V2 integration | `GitHubManager` | GitHub sync workflows |
