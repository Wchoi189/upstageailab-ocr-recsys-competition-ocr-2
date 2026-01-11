# Phase 2: Context Bundling — Quick Implementation Checklist

**Estimated Time**: 2-3 hours
**Status**: Ready to implement after Phase 1 complete

---

## 🎯 Quick Summary

**What**: Integrate utility scripts discovery into AgentQMS context bundling system
**Why**: Auto-suggest utilities when agents ask relevant questions
**How**: Create bundle definition YAML that plugs into existing system
**Result**: Agents see utility suggestions automatically (no manual action needed)

---

## ✅ Implementation Checklist

### Step 1: Create Bundle Definition (20 mins)

```bash
# Create directory
mkdir -p .agentqms/plugins/context_bundles

# Create bundle definition file
```

**File**: `.agentqms/plugins/context_bundles/utility-scripts.yaml`

```yaml
name: utility-scripts
title: "Reusable Utility Scripts Discovery"
description: |
  Discover and use reusable utility scripts instead of writing custom code.
  Utilities: ConfigLoader (caching), paths, timestamps, git.

ads_version: "1.0"
scope: "project"

tags:
  - utilities
  - discovery
  - configuration
  - caching

triggers:
  keywords:
    # Configuration
    - "load yaml"
    - "load config"
    - "configuration"
    - "config file"
    # Caching
    - "cache"
    - "caching"
    - "performance"
    # Paths
    - "project root"
    - "artifacts directory"
    - "artifact location"
    - "find path"
    - "get path"
    # Timestamps
    - "timestamp"
    - "kst"
    - "timezone"
    - "current time"
    # Git
    - "current branch"
    - "git branch"
    - "commit hash"

  patterns:
    - "\\b(yaml|config)\\b.*\\b(load|import|parse)\\b"
    - "\\b(load|import)\\b.*\\b(yaml|config)\\b"
    - "\\b(path|directory)\\b.*\\b(find|get|resolve)\\b"
    - "\\b(timestamp|date|time)\\b.*\\b(format|current)\\b"
    - "\\b(branch|commit)\\b.*\\b(current|detect)\\b"

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
    description: "Use-case guides"
    priority: "normal"
    files:
      - "context/utility-scripts/by-use-case/"
```

**Status**: ✅ Create this file

---

### Step 2: Verify Plugin Loading (10 mins)

Test that the bundle loads correctly:

```bash
# Test 1: Check file exists
ls -la .agentqms/plugins/context_bundles/utility-scripts.yaml

# Test 2: Verify YAML syntax
python3 -c "import yaml; yaml.safe_load(open('.agentqms/plugins/context_bundles/utility-scripts.yaml'))"

# Test 3: Check bundle is discoverable
python3 AgentQMS/tools/utilities/suggest_context.py "load yaml config"
# Should output: utility-scripts bundle suggested with high score
```

**Expected Output**:
```
Bundle: utility-scripts
Score: 0.95 (very high match)
Tiers: tier1, tier2 included
Files: QUICK_REFERENCE.md, UTILITY_SCRIPTS_INDEX.yaml, config_loader.md
```

**Status**: ⏳ Test after Phase 1 complete

---

### Step 3: Update Agent Instructions (20 mins)

**File**: `.github/copilot-instructions.md`

Add this section:

```markdown
## Reusable Utilities: Discover Before You Code

AgentQMS provides reusable utility scripts to avoid reinventing the wheel.

### Most Useful Utilities

| Task | Utility | Why Use It |
|------|---------|-----------|
| Load YAML config | `ConfigLoader` | Auto-caching, ~2000x faster on repeats |
| Find paths (root, artifacts, docs) | `paths.get_project_root()` etc. | No hardcoded paths, project-aware |
| Current timestamp | `timestamps.get_kst_timestamp()` | Consistent KST formatting |
| Current git branch | `git.get_current_branch()` | Clean API, graceful fallbacks |

### How to Discover

The system will **automatically suggest utilities** when you ask about:
- Loading configuration files
- Finding project directories
- Handling timestamps
- Git information

**Or manually**:
```bash
python AgentQMS/tools/utilities/suggest_context.py "your task"
```

### Quick Reference

See: `context/utility-scripts/QUICK_REFERENCE.md`

### Key Insight

**ConfigLoader** provides automatic LRU caching. Using it instead of custom
`yaml.safe_load()` gives ~2000x performance improvement on repeated access.
```

**Status**: ✅ Ready to add

---

### Step 4: Create Integration Documentation (20 mins)

**File**: `context/utility-scripts/PHASE_2_INTEGRATION.md`

```markdown
# Phase 2: Context Bundling Integration

## How It Works

This bundle is now integrated with AgentQMS context system.

When you mention relevant keywords, the system suggests utilities:

**Triggers** (any of these will suggest the bundle):
- "load yaml" → ConfigLoader
- "project root" → paths utilities
- "timestamp" → timestamps utilities
- "current branch" → git utilities
- Many more keywords...

## Example Scenarios

**Scenario 1: Load Configuration**
```
You: "Load a YAML config file with defaults"
System: Suggests utility-scripts bundle
  → ConfigLoader with examples
  → 2000x caching benefit highlighted
```

**Scenario 2: Find Paths**
```
You: "Where should artifact files go?"
System: Suggests utility-scripts bundle
  → paths.get_artifacts_dir()
  → No hardcoding needed
```

**Scenario 3: Multiple Utilities**
```
You: "Load config, find artifacts, timestamp them"
System: Suggests utility-scripts bundle
  → All 3 utilities included
  → Full workflow with examples
```

## Testing

Test that suggestions work:

```bash
python AgentQMS/tools/utilities/suggest_context.py \
    "load yaml config file"
# Should suggest utility-scripts bundle

python AgentQMS/tools/utilities/suggest_context.py \
    "find project root and artifacts"
# Should suggest utility-scripts bundle

python AgentQMS/tools/utilities/suggest_context.py \
    "design neural network"
# Should NOT suggest utility-scripts (wrong context)
```

## Tiers

Bundle is organized for progressive disclosure:

- **Tier 1**: Quick reference + index (always loaded)
- **Tier 2**: Detailed docs for 4 top utilities (usually loaded)
- **Tier 3**: Use-case guides (loaded if very high relevance)

This keeps context focused and relevant.

## Integration Points

- **suggest_context.py**: Auto-detects utility-relevant tasks
- **Plugin Registry**: Discovers and loads bundle definition
- **Context System**: Injects bundle into conversations
- **Agent Instructions**: References utility discovery

## Support

For full details, see:
- **Analysis**: `/analysis/PHASE_2_CONTEXT_BUNDLING_PLAN.md`
- **Quick Ref**: `context/utility-scripts/QUICK_REFERENCE.md`
- **Index**: `context/utility-scripts/UTILITY_SCRIPTS_INDEX.yaml`
```

**Status**: ✅ Ready to create

---

### Step 5: Comprehensive Testing (30 mins)

**Test Case 1: Config Loading**
```bash
python AgentQMS/tools/utilities/suggest_context.py \
    "I need to load YAML configuration file"

Expected:
✓ Bundle suggested
✓ Score > 0.8
✓ ConfigLoader docs included
```

**Test Case 2: Path Resolution**
```bash
python AgentQMS/tools/utilities/suggest_context.py \
    "find the artifacts directory in the project"

Expected:
✓ Bundle suggested
✓ Score > 0.7
✓ paths.md included
✓ ConfigLoader.md NOT needed
```

**Test Case 3: Git Integration**
```bash
python AgentQMS/tools/utilities/suggest_context.py \
    "detect current git branch for artifact metadata"

Expected:
✓ Bundle suggested
✓ Score > 0.7
✓ git.md included
```

**Test Case 4: Multiple Utilities**
```bash
python AgentQMS/tools/utilities/suggest_context.py \
    "Load config, find artifacts dir, timestamp them, get branch"

Expected:
✓ Bundle suggested
✓ Score > 0.9 (very high)
✓ All utilities included
```

**Test Case 5: Non-Matching Task**
```bash
python AgentQMS/tools/utilities/suggest_context.py \
    "design a deep learning architecture for OCR"

Expected:
✓ Bundle NOT suggested (score < 0.3)
✓ Other bundles may be suggested instead
```

**Status**: ⏳ Execute after Phase 1

---

### Step 6: Documentation & Handoff (20 mins)

Create final summary:

```markdown
# Phase 2 Complete ✅

## What Was Done

1. ✅ Created `.agentqms/plugins/context_bundles/utility-scripts.yaml`
   - Comprehensive trigger keywords
   - Regex patterns for precision
   - Tier-based file organization

2. ✅ Verified integration with `suggest_context.py`
   - Bundle loads without errors
   - Suggestions work correctly
   - All test cases pass

3. ✅ Updated agent instructions
   - Added utility discovery section
   - Provided quick reference
   - Highlighted performance benefits

4. ✅ Created integration documentation
   - Explained how it works
   - Provided test procedures
   - Documented trigger keywords

## Results

- Utilities now **auto-discovered** in relevant conversations
- Agents see suggestions when asking about:
  - Configuration loading
  - Path resolution
  - Timestamps
  - Git information
  - Caching/performance

- **No manual action needed** — context system handles it

## What's Next?

### Optional Phase 3: MCP Tool
Create `list_utilities()` tool for programmatic queries

### Optional Phase 4: Auto-Generation
Auto-scan utilities, self-maintaining index

## Success Metrics

- ✅ Bundle loads successfully
- ✅ Suggestions triggered on relevant tasks
- ✅ No false positives (non-relevant tasks don't trigger)
- ✅ Documentation clear and findable
- ✅ Agent instructions updated
```

**Status**: ✅ Create after all tests pass

---

## 📊 Phase 2 Timeline

```
Step 1: Bundle Definition       20 mins
Step 2: Plugin Loading Test     10 mins
Step 3: Agent Instructions      20 mins
Step 4: Integration Docs        20 mins
Step 5: Comprehensive Testing   30 mins
Step 6: Documentation & Review  20 mins
────────────────────────────────────────
TOTAL:                          2h 20 mins
```

**With breaks/buffer**: 2.5-3 hours

---

## 🚀 Quick Start Command

Once Phase 1 is complete, run this to start Phase 2:

```bash
# Create bundle directory
mkdir -p .agentqms/plugins/context_bundles

# Create bundle definition (copy template above to file)
cat > .agentqms/plugins/context_bundles/utility-scripts.yaml << 'EOF'
[... bundle YAML from Step 1 above ...]
EOF

# Test it works
python AgentQMS/tools/utilities/suggest_context.py "load yaml config"

# If tests pass, Phase 2 is done!
```

---

## 📋 Checklist Summary

- [ ] **Create** `.agentqms/plugins/context_bundles/utility-scripts.yaml`
- [ ] **Test** bundle loads with `suggest_context.py`
- [ ] **Verify** test cases 1-5 pass
- [ ] **Update** `.github/copilot-instructions.md`
- [ ] **Create** `context/utility-scripts/PHASE_2_INTEGRATION.md`
- [ ] **Document** testing procedures
- [ ] **Verify** no breaking changes
- [ ] **Mark Phase 2 complete** ✅

---

## 💡 Key Points

1. **No code changes needed** — Uses existing plugin system
2. **Completely non-breaking** — Just adds a new bundle
3. **Backward compatible** — Existing bundles still work
4. **Fast to implement** — 2-3 hours total
5. **High value** — Auto-suggests utilities (agents see them)

---

## Support

Questions during Phase 2 implementation?

1. **Bundle definition syntax**: See `PHASE_2_CONTEXT_BUNDLING_PLAN.md`
2. **How triggers work**: Check `suggest_context.py` source
3. **Testing procedures**: Use test cases in Step 5
4. **Integration**: Reference PHASE_2_INTEGRATION.md

---

**Ready to implement?** Start with Phase 1, then Phase 2 when ready! 🎉
