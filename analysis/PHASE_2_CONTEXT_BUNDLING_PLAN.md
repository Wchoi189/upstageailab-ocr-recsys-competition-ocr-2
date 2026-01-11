# Phase 2: Context Bundling Integration — Implementation Plan

**Status**: Detailed design for integration with existing context system
**Date**: 2026-01-11
**Target Completion**: 2-3 hours after Phase 1 complete

---

## 1. Overview: What Is Phase 2?

Phase 2 integrates the utility scripts discovery system into AgentQMS's existing **context bundling framework**.

Instead of agents manually reading documentation, utilities will be **automatically suggested in relevant conversations** based on task keywords.

### How It Works

```
Agent Task: "Load YAML configuration"
        ↓
[Context System Analyzes Task]
        ↓
[Matches Keywords: "load", "yaml", "config"]
        ↓
[Triggers: utility-scripts bundle]
        ↓
[Injects into conversation]
        ↓
Agent sees: "Consider using ConfigLoader utility"
        ↓
Agent uses ConfigLoader instead of custom code
```

---

## 2. Existing Infrastructure You Can Leverage

### 2.1 Current Context System

**Location**: `AgentQMS/tools/utilities/suggest_context.py`

**What it does**:
- Analyzes task descriptions
- Matches against keywords and patterns
- Suggests appropriate context bundles
- Supports regex-based pattern matching

**Key class**: `ContextSuggester`
- Method: `get_suggested_bundles(task_description)`
- Returns: List of recommended bundles with scores

### 2.2 Bundle Schema

**File**: `AgentQMS/standards/schemas/plugin_context_bundle.json`

**Structure**:
```json
{
    "name": "utility-scripts",
    "title": "Reusable Utility Scripts",
    "tiers": {
        "tier1": {
            "files": ["QUICK_REFERENCE.md", "UTILITY_SCRIPTS_INDEX.yaml"],
            "priority": "critical"
        },
        "tier2": {
            "files": ["by-category/*.md"],
            "priority": "high"
        },
        "tier3": {
            "files": ["by-use-case/*.md"],
            "priority": "normal"
        }
    },
    "triggers": {
        "keywords": [
            "load yaml", "load config", "configuration",
            "project root", "artifacts directory", "paths",
            "timestamp", "current branch", "git",
            "cache", "caching", "performance"
        ],
        "patterns": [
            "\\b(yaml|config)\\b.*\\b(load|import)\\b",
            "\\b(path|directory|location)\\b.*\\b(find|get|resolve)\\b",
            "\\b(timestamp|date|time)\\b.*\\b(format|current)\\b",
            "\\b(branch|commit|git)\\b.*\\b(detect|current)\\b"
        ]
    }
}
```

### 2.3 Plugin Registry

**Location**: `.agentqms/plugins/`

**How bundles are discovered**:
1. YAML files in plugin registry define bundles
2. `ContextSuggester` loads bundles from registry
3. Task descriptions matched against triggers
4. Top-scoring bundles recommended

---

## 3. Phase 2 Implementation Steps

### Step 1: Create Bundle Definition (20 mins)

**File**: `.agentqms/plugins/context_bundles/utility-scripts.yaml`

```yaml
name: utility-scripts
title: "Reusable Utility Scripts Discovery"
description: |
  Discover and use reusable utility scripts instead of writing custom code.
  Includes YAML loading (with caching), path resolution, timestamps, git utilities.

ads_version: "1.0"
scope: "project"

tags:
  - utilities
  - discovery
  - configuration
  - paths
  - timestamps
  - git
  - reuse

triggers:
  keywords:
    # Configuration loading
    - "load yaml"
    - "load config"
    - "configuration"
    - "config file"
    - "yaml file"
    # Caching keywords
    - "caching"
    - "cache"
    - "performance"
    # Path resolution
    - "project root"
    - "artifacts directory"
    - "artifacts dir"
    - "artifact location"
    - "standard path"
    - "standard directory"
    - "find path"
    - "get path"
    - "locate file"
    # Timestamps
    - "timestamp"
    - "kst"
    - "timezone"
    - "current time"
    - "format date"
    - "date format"
    # Git utilities
    - "current branch"
    - "git branch"
    - "commit hash"
    - "git commit"
    - "detect branch"

  patterns:
    # YAML/Config patterns
    - "\\b(yaml|config)\\b.*\\b(load|import|parse)\\b"
    - "\\b(load|import|parse)\\b.*\\b(yaml|config|configuration)\\b"

    # Path patterns
    - "\\b(path|directory|location|file)\\b.*\\b(find|get|resolve|locate)\\b"
    - "\\b(project|artifact|document)\\b.*\\b(root|dir|location)\\b"

    # Timestamp patterns
    - "\\b(timestamp|datetime|date|time)\\b.*\\b(format|current|kst)\\b"
    - "\\b(format|create)\\b.*\\b(timestamp|date|time)\\b"

    # Git patterns
    - "\\b(branch|commit|git)\\b.*\\b(current|detect|get)\\b"
    - "\\b(current|get)\\b.*\\b(branch|commit)\\b"

    # Caching patterns
    - "\\b(cache|caching|performance)\\b.*\\b(load|config|yaml)\\b"

tiers:
  tier1:
    description: "Critical discovery resources"
    priority: "critical"
    files:
      - "context/utility-scripts/QUICK_REFERENCE.md"
      - "context/utility-scripts/UTILITY_SCRIPTS_INDEX.yaml"

  tier2:
    description: "Detailed utility documentation"
    priority: "high"
    files:
      - "context/utility-scripts/by-category/configuration/config_loader.md"
      - "context/utility-scripts/by-category/path-resolution/paths.md"
      - "context/utility-scripts/by-category/timestamps/timestamps.md"
      - "context/utility-scripts/by-category/git/git.md"

  tier3:
    description: "Use-case based guides"
    priority: "normal"
    files:
      - "context/utility-scripts/by-use-case/"

hints:
  # When each utility is most useful
  - |
    **ConfigLoader**: Use when you need to load YAML configuration files.
    Benefits: Automatic LRU caching (~2000x speedup), graceful fallbacks, type-safe.

  - |
    **paths**: Use when you need to find standard project directories.
    Benefits: No hardcoded paths, consistent across codebase, project-root-aware.

  - |
    **timestamps**: Use when you need to handle timestamps for artifact metadata.
    Benefits: KST timezone handling, format consistency, age calculations.

  - |
    **git**: Use when you need git information (current branch, commit hash).
    Benefits: Graceful fallbacks, subprocess-free, cached results.
```

### Step 2: Integrate with ContextSuggester (30 mins)

The existing `suggest_context.py` should auto-pick up the bundle because:

1. ✅ Plugin registry loads bundles automatically
2. ✅ Bundle triggers (keywords + patterns) are already supported
3. ✅ Scoring mechanism already exists
4. ✅ **No code changes needed** (plugin-based)

**Verification**:
```bash
python AgentQMS/tools/utilities/suggest_context.py "load yaml config"
# Should output: utility-scripts bundle suggested

python AgentQMS/tools/utilities/suggest_context.py "find project root"
# Should output: utility-scripts bundle suggested
```

### Step 3: Create Agent Instructions Addition (20 mins)

**File to update**: `.github/copilot-instructions.md` or agent prompt

**Add section**:
```markdown
## Reusable Utilities: Utility Scripts Bundle

Before writing custom code for these tasks, check if a utility exists:

**Available Utilities**:
- `config_loader` — Load YAML with caching (~2000x faster)
- `paths` — Resolve standard project paths
- `timestamps` — Handle KST timestamps
- `git` — Detect git branch/commit info

**Quick Discovery**:
- Use context bundling: System will suggest utilities automatically
- Or run: `python AgentQMS/tools/utilities/suggest_context.py "<task>"`
- Or read: `context/utility-scripts/QUICK_REFERENCE.md`

**Key Insight**: ConfigLoader provides automatic caching.
Using it instead of custom `yaml.safe_load()` = ~2000x performance gain.
```

### Step 4: Test Integration (30 mins)

Test the context system suggests utilities:

```bash
# Test 1: Config loading suggestion
python AgentQMS/tools/utilities/suggest_context.py \
    "I need to load a YAML configuration file with fallback defaults"

# Expected output: utility-scripts bundle with high score

# Test 2: Path resolution suggestion
python AgentQMS/tools/utilities/suggest_context.py \
    "Where should I look for artifact files in the project?"

# Expected output: utility-scripts bundle with high score

# Test 3: Timestamp suggestion
python AgentQMS/tools/utilities/suggest_context.py \
    "Create a KST timestamp for artifact metadata"

# Expected output: utility-scripts bundle with high score
```

### Step 5: Document Phase 2 (20 mins)

**File**: `context/utility-scripts/PHASE_2_INTEGRATION.md`

```markdown
# Phase 2: Context Bundling Integration

This bundle is integrated with AgentQMS context bundling system.

## How It Works

When you ask AgentQMS for help with:
- Loading configuration files
- Resolving project paths
- Handling timestamps
- Git branch/commit detection

The system automatically suggests the **utility-scripts** bundle.

## Triggering Suggestions

### Automatic (In Conversation)
Just mention a relevant keyword:
- "load YAML config" → suggests ConfigLoader
- "find project root" → suggests paths utilities
- "current timestamp" → suggests timestamps utilities
- "detect branch" → suggests git utilities

### Manual (Command Line)
```bash
python AgentQMS/tools/utilities/suggest_context.py "your task here"
```

### Manual (Direct)
Read the bundle directly:
- Quick reference: `context/utility-scripts/QUICK_REFERENCE.md`
- Full index: `context/utility-scripts/UTILITY_SCRIPTS_INDEX.yaml`

## Tiers

The bundle is organized in 3 tiers:

1. **Tier 1 (Critical)**: QUICK_REFERENCE.md, Index
   - Always included when bundle suggested

2. **Tier 2 (High)**: Detailed docs for top 4 utilities
   - Included for high-relevance suggestions

3. **Tier 3 (Normal)**: Use-case guides
   - Included for contextual learning

## Integration Points

- **suggest_context.py**: Auto-detects when utilities might help
- **MCP Servers**: Can query bundle for documentation
- **Agent Instructions**: References the bundle
- **Context System**: Bundles with other resources as needed
```

---

## 4. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│  Agent Task Description                                  │
│  "Load YAML config file with default fallback"          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  ContextSuggester.get_suggested_bundles()               │
│  1. Parse task keywords                                 │
│  2. Match against bundle triggers                       │
│  3. Score matches                                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Plugin Registry Loads: utility-scripts.yaml             │
│  Checks:                                                 │
│  - Keywords: ["load yaml", "load config", ...]          │
│  - Patterns: [r"yaml.*load", ...]                       │
│  Score: 0.95 (high match!)                              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Bundle Files Loaded (Tiers 1 & 2)                       │
│  - QUICK_REFERENCE.md                                   │
│  - UTILITY_SCRIPTS_INDEX.yaml                           │
│  - config_loader.md                                     │
│  - Other utility docs                                   │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Agent Receives Suggestion                               │
│  "Consider using ConfigLoader utility"                  │
│  [With documentation & examples]                        │
└──────────────────────────────────────────────────────────┘
```

---

## 5. Integration with Existing Systems

### 5.1 How It Fits with Phase 1

**Phase 1 Creates**:
- `context/utility-scripts/` directory
- QUICK_REFERENCE.md
- UTILITY_SCRIPTS_INDEX.yaml
- Detailed markdown files

**Phase 2 Uses**:
- All Phase 1 files
- Creates `.agentqms/plugins/context_bundles/utility-scripts.yaml`
- Integrates with existing `suggest_context.py`
- No modifications to core systems needed

### 5.2 Compatibility

✅ Works with existing context system
✅ Uses established plugin architecture
✅ Leverages current bundle schema
✅ No breaking changes
✅ Backward compatible

### 5.3 Integration Points

```
Agent                    Existing Systems              Phase 2
├─ Task description      └─ suggest_context.py ◄──────┘ (triggers)
└─ Asks for help        └─ Plugin Registry   ◄──────┘ (loads bundle)
                        └─ Context Bundling ◄──────┘ (injects docs)
                        └─ Agent Instructions ◄──┘ (references)
```

---

## 6. Configuration Parameters

### 6.1 Trigger Matching Sensitivity

**Current Plan**: Balanced triggers
- ~30 keywords for high recall
- ~6 regex patterns for precision
- Expected match rate: 70-80% of relevant tasks

**If needed to tune**:
- Add more keywords → higher recall, more false positives
- Add more patterns → precision matching
- Adjust keyword weights in scorer

### 6.2 Tier Inclusion Thresholds

**Current Plan**:
- Tier 1 (Critical): Always included (score > 0)
- Tier 2 (High): Included if score > 0.5
- Tier 3 (Normal): Included if score > 0.7

**Tunable**: In bundle definition

---

## 7. Phase 2 Deliverables Checklist

- [ ] Create `.agentqms/plugins/context_bundles/utility-scripts.yaml`
  - [ ] Define bundle metadata
  - [ ] Add comprehensive triggers (keywords + patterns)
  - [ ] Define tiers and files
  - [ ] Add helpful hints

- [ ] Create `context/utility-scripts/PHASE_2_INTEGRATION.md`
  - [ ] Explain how it works
  - [ ] Document trigger keywords
  - [ ] Show integration points
  - [ ] Provide testing instructions

- [ ] Update `.github/copilot-instructions.md`
  - [ ] Add "Reusable Utilities" section
  - [ ] Reference utility-scripts bundle
  - [ ] Show quick reference table

- [ ] Verify Integration
  - [ ] Run `suggest_context.py` with test queries
  - [ ] Confirm utility-scripts bundle suggested
  - [ ] Check tier loading works correctly

- [ ] Documentation
  - [ ] Document the bundle definition
  - [ ] Explain trigger system
  - [ ] Show example scenarios

---

## 8. Testing Strategy

### Test Cases

**Test 1: Config Loading**
```bash
$ python AgentQMS/tools/utilities/suggest_context.py \
    "I need to load YAML configuration with defaults"

Expected Output:
- utility-scripts bundle suggested
- Score: ~0.9 (high relevance)
- Tiers 1 & 2 included
```

**Test 2: Path Resolution**
```bash
$ python AgentQMS/tools/utilities/suggest_context.py \
    "Where do I find the artifacts directory?"

Expected Output:
- utility-scripts bundle suggested
- Score: ~0.8
- ConfigLoader.md not included (not relevant)
- paths.md included
```

**Test 3: Multiple Utilities**
```bash
$ python AgentQMS/tools/utilities/suggest_context.py \
    "Load config and find artifact paths with current timestamp"

Expected Output:
- utility-scripts bundle suggested
- Score: ~0.95 (very high)
- All relevant docs included
```

**Test 4: Non-Matching Task**
```bash
$ python AgentQMS/tools/utilities/suggest_context.py \
    "Design a new neural network architecture"

Expected Output:
- utility-scripts bundle NOT suggested (or very low score)
- Other relevant bundles suggested instead
```

---

## 9. Success Criteria

### Metrics

| Criterion | Target | How to Measure |
|-----------|--------|----------------|
| Trigger Accuracy | 90%+ relevant tasks trigger bundle | Manual testing |
| False Positives | <10% incorrect suggestions | Test with non-matching tasks |
| Integration | Bundle loads without errors | Verify with tests |
| Discoverability | Agents find utilities via bundle | Usage tracking |

### Example Success Scenario

```
Agent asks: "Create an artifact with YAML metadata config"

System responds:
"Consider using these utilities:
- ConfigLoader: Load YAML with caching (~2000x faster)
- paths: Find standard directories
- timestamps: Format timestamps"

Agent uses ConfigLoader instead of custom yaml.safe_load()
Result: Automatic caching, better performance, consistent APIs
```

---

## 10. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Bundle not triggered | Low | Comprehensive trigger keywords |
| False positives | Low | Regex patterns for precision |
| Context overload | Low | Tier-based loading (only relevant docs) |
| Plugin not found | Medium | Explicit path in bundle definition |

---

## 11. Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Phase 1 | 2-4h | Documentation setup |
| Phase 2 | 2-3h | Bundle definition + integration |
| **Total** | **4-7h** | **Full discovery system** |

**Phase 2 Breakdown**:
- Bundle definition: 20 mins
- Integration check: 10 mins
- Agent instructions: 20 mins
- Testing: 30 mins
- Documentation: 20 mins
- Buffer: 30 mins
- **Total**: ~2.5 hours

---

## 12. After Phase 2 (Optional Enhancements)

### Phase 3 Enhancement: MCP Tool
Add `list_utilities()` tool for programmatic access

### Phase 4 Enhancement: Auto-Generation
Auto-scan utilities, self-maintaining index

### Future: Analytics
Track which utilities agents discover/use most

---

## Appendix: Bundle Definition Template

```yaml
# Minimal Phase 2 bundle definition
name: utility-scripts
title: "Reusable Utility Scripts"

triggers:
  keywords:
    - "load yaml"
    - "load config"
    - "project root"
    - "timestamp"
    - "current branch"

tiers:
  tier1:
    files:
      - "context/utility-scripts/QUICK_REFERENCE.md"
      - "context/utility-scripts/UTILITY_SCRIPTS_INDEX.yaml"
```

This minimal bundle will work immediately and can be enhanced over time.

---

## Next Steps

1. **Complete Phase 1** (Documentation)
   - Set up directory structure
   - Write QUICK_REFERENCE.md
   - Create YAML index

2. **Begin Phase 2** (when ready, 2-3 hours later)
   - Create bundle definition YAML
   - Verify with tests
   - Update instructions

3. **Monitor & Iterate**
   - Track suggestion accuracy
   - Adjust triggers as needed
   - Enhance based on agent feedback

---

**Ready to implement Phase 2?**
Just let me know when Phase 1 is complete! 🚀
