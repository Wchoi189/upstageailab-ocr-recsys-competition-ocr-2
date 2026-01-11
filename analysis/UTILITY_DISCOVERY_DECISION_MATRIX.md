# Utility Discovery Implementation Strategy — Decision Matrix

## Overview

Three viable approaches exist. This matrix helps choose based on your priorities.

---

## Approach Comparison

### Approach 1: Documentation-Driven (Recommended for Phase 1)

**What**: Create markdown docs + YAML index in `context/utility-scripts/`

**How It Works**:
1. Create structured documentation for each utility
2. Organize by category and use-case
3. Create QUICK_REFERENCE.md for quick lookup
4. Include in agent prompt/context bundling

**Implementation**:
```
context/utility-scripts/
├── UTILITY_SCRIPTS_INDEX.yaml     (searchable index)
├── QUICK_REFERENCE.md             (TL;DR)
├── by-category/
│   ├── configuration/
│   │   └── config_loader.md       (detailed guide)
│   ├── path-resolution/
│   │   └── paths.md
│   └── ...
└── by-use-case/
    ├── "I need to load config"
    ├── "I need to find a path"
    └── ...
```

**Time to Implement**: 2-4 hours

**Pros**:
- ✅ Immediate discoverability
- ✅ Detailed, hand-curated examples
- ✅ Easy to update with gotchas/patterns
- ✅ Works with current context system
- ✅ Searchable by humans and AI
- ✅ Low technical overhead

**Cons**:
- ❌ Manual maintenance (doc drift over time)
- ❌ Requires manual updates when utilities change
- ❌ Relies on agent reading docs (not guaranteed)
- ❌ No programmatic interface

**Best For**:
- Immediate discoverability
- Detailed pattern guidance
- Legacy/stable utilities
- Getting started quickly

**Example**: Agent reads QUICK_REFERENCE.md, sees ConfigLoader suggestion, uses it.

---

### Approach 2: MCP Tool (Recommended for Phase 3)

**What**: Create `list_utilities()` tool that agents can query programmatically

**How It Works**:
1. Add tool to unified_server.py or mcp_server.py
2. Tool searches index, filters by criteria
3. Agent calls tool when needed
4. Tool returns utility info + examples

**Implementation**:
```python
@app.list_tools()
async def list_utilities():
    return [Tool(
        name="list_utilities",
        description="Find reusable utility scripts by purpose/category",
        inputSchema={
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "desc": "What do you need?"},
                "category": {"type": "string", "desc": "Category to search"},
                "keywords": {"type": "array", "items": {"type": "string"}}
            }
        }
    )]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "list_utilities":
        return search_utilities(
            purpose=arguments.get("purpose"),
            category=arguments.get("category"),
            keywords=arguments.get("keywords")
        )
```

**Time to Implement**: 3-5 hours

**Pros**:
- ✅ Explicit, programmable interface
- ✅ Agent controls when to search
- ✅ Can return filtered, contextual results
- ✅ Enables advanced queries
- ✅ Programmatic access (non-agent code too)
- ✅ Scalable to many utilities

**Cons**:
- ❌ Requires agent to think to call tool
- ❌ Tool overhead (latency)
- ❌ Still needs underlying documentation
- ❌ More code to maintain
- ❌ Requires MCP server updates

**Best For**:
- Programmatic discovery
- Large utility sets
- Advanced filtering
- Automation workflows

**Example**:
```
Agent: "I need to load YAML config"
→ Agent calls: list_utilities(purpose="load config")
→ Tool returns: ConfigLoader with 5 usage examples
→ Agent uses it
```

---

### Approach 3: Auto-Generation (Recommended for Phase 4+)

**What**: Scan source code at startup, extract docstrings/examples, auto-build index

**How It Works**:
1. Scan `AgentQMS/tools/utils/*.py` at startup
2. Extract classes, functions, docstrings via AST
3. Build searchable index dynamically
4. Watch for file changes, rebuild on change
5. Combine with hand-written guidance docs

**Implementation**:
```python
class UtilityIndexer:
    def scan_utilities(self, utils_dir: Path) -> dict:
        """Scan utils directory, extract metadata."""
        index = {}
        for py_file in utils_dir.glob("*.py"):
            module = ast.parse(py_file.read_text())
            # Extract classes, functions, docstrings
            # Build index entries
        return index

    def watch_for_changes(self):
        """Watch directory, rebuild on changes."""
        # Use watchdog or similar
```

**Time to Implement**: 4-8 hours

**Pros**:
- ✅ Always in sync with source code
- ✅ Captures latest APIs automatically
- ✅ No manual documentation drift
- ✅ Extracts real examples from docstrings
- ✅ Minimal maintenance burden
- ✅ Self-documenting

**Cons**:
- ❌ Complex implementation
- ❌ Requires AST parsing, watching
- ❌ Can miss context/patterns (docstrings aren't enough)
- ❌ Hand-written guidance still needed
- ❌ More moving parts to maintain
- ❌ Startup overhead

**Best For**:
- Large, rapidly evolving utility sets
- Long-term maintainability
- Automated pipelines
- When manual docs can't keep up

**Example**:
```
Startup: Scan AgentQMS/tools/utils/ → Extract 47 functions → Build index
Runtime: Agent queries → Index returns fresh results
Update: Dev adds new utility → Auto-detected on next scan
```

---

## Decision Framework

### Choose Based on These Factors

**Factor 1: Time to Deliver Value**
- Documentation: **1-2 weeks** (Phase 1 now)
- MCP Tool: **3-4 weeks** (Phase 2 after Phase 1)
- Auto-Gen: **2-3 months** (Phase 4, needs investigation)

**Factor 2: Maintenance Burden**
- Documentation: **Medium** (manual updates per release)
- MCP Tool: **Medium** (tool + docs to maintain)
- Auto-Gen: **Low** (mostly self-maintaining)

**Factor 3: Current Utility Count**
- 7 utilities → Documentation is fine
- 20+ utilities → Consider MCP Tool
- 50+ utilities → Auto-Gen becomes attractive

**Factor 4: Utility Stability**
- Stable utilities → Documentation works
- Frequently changing → Auto-Gen better
- Mix of both → Hybrid approach

**Factor 5: Budget/Effort**
- 4 hours → Documentation only
- 8 hours → Documentation + MCP Tool
- 20+ hours → Full hybrid system

---

## Recommended Roadmap

### Stage 1: Immediate (This Week) — Documentation
**Effort**: 2-4 hours
**Impact**: High immediate value
**What**:
- Create `context/utility-scripts/` structure
- Write markdown for top 3 utilities (config_loader, paths, timestamps)
- Create QUICK_REFERENCE.md
- Create UTILITY_SCRIPTS_INDEX.yaml
- Include in agent prompts

**Deliverable**: Agents can discover utilities via documentation

**Success Metric**: Agent uses ConfigLoader instead of custom yaml.safe_load()

---

### Stage 2: Phase 2 (Next Month) — Context Integration
**Effort**: 2-3 hours
**Impact**: Seamless discovery
**What**:
- Bundle utility-scripts into context system
- Trigger on relevant keywords ("config", "path", "timestamp", etc.)
- Test with sample agent queries

**Deliverable**: Utilities auto-injected into relevant conversations

**Success Metric**: Agent discovers utilities without being asked

---

### Stage 3: Phase 3 (Later) — MPC Tool [Optional]
**Effort**: 3-5 hours
**Impact**: Programmatic access
**What**:
- Add `list_utilities()` tool to MCP server
- Implement search/filter logic
- Document tool usage

**Deliverable**: Agents can query utilities like any other tool

**Success Metric**: Tools called in relevant contexts

---

### Stage 4: Phase 4 [Future] — Auto-Generation
**Effort**: 6-10 hours
**Impact**: Long-term maintainability
**What**:
- Implement UtilityIndexer class
- Scan source code at startup
- Watch for changes
- Combine with hand-written guidance

**Deliverable**: Self-maintaining index

**Success Metric**: Zero documentation drift

---

## Recommended Starting Point

**Best Option**: **Documentation + Phase 2 Context**

**Why**:
1. ✅ Quick to implement (Phase 1: 2-4h)
2. ✅ Immediate value (discovery works today)
3. ✅ Low maintenance (manual docs are stable)
4. ✅ Future-proof (can add MCP Tool later)
5. ✅ Works with current systems

**Not Recommended Yet**: Auto-generation
- **Why**: Overkill for 7 utilities
- **When**: Revisit when utility count hits 30+

---

## Implementation Checklist for Phase 1

- [ ] Create directory: `context/utility-scripts/`
- [ ] Create UTILITY_SCRIPTS_INDEX.yaml (index file)
- [ ] Create QUICK_REFERENCE.md (one-pager)
- [ ] Create by-category/ subdirectories:
  - [ ] configuration/
  - [ ] path-resolution/
  - [ ] timestamps/
  - [ ] git/
  - [ ] github/ (optional for Phase 1)
- [ ] Write detailed markdown files:
  - [ ] config_loader.md
  - [ ] paths.md
  - [ ] timestamps.md
  - [ ] git.md (optional for Phase 1)
- [ ] Create by-use-case/ examples
- [ ] Add reference to agent instructions
- [ ] Test with sample agent query: "I need to load a YAML config"

**Estimated Time**: 3-4 hours
**Value Delivered**: Immediate discoverability

---

## Sample QUICK_REFERENCE.md

```markdown
# Reusable Utilities — Quick Reference

## Common Tasks

| Task | Utility | Example |
|------|---------|---------|
| Load YAML config | `config_loader.ConfigLoader` | `ConfigLoader.load_yaml("config.yaml")` |
| Get project root | `paths.get_project_root()` | `root = get_project_root()` |
| Get artifact dir | `paths.get_artifacts_dir()` | `artifacts = get_artifacts_dir()` |
| Get current branch | `git.get_current_branch()` | `branch = get_current_branch()` |
| Get timestamp | `timestamps.get_kst_timestamp()` | `ts = get_kst_timestamp()` |

## Full Index

See: `UTILITY_SCRIPTS_INDEX.yaml` or browse `by-category/`

## Need Help?

- Task-based lookup: See `by-use-case/`
- Category-based: Browse `by-category/`
- Detailed guide: Read `.md` file for utility

## How to Contribute

Adding a new utility?
1. Document in `by-category/{category}/`
2. Add entry to UTILITY_SCRIPTS_INDEX.yaml
3. Update QUICK_REFERENCE.md if high-value
```

---

## Final Recommendation

**Start with Phase 1 Documentation (2-4 hours)**
- Immediate, tangible value
- Sets foundation for future phases
- Low risk, easy to iterate
- Can evaluate real impact before Phase 2

**Then decide on Phase 2+ based on results**
- If documentation works → add Phase 2 Context
- If agents still don't discover → add Phase 3 MCP Tool
- If utility count grows → plan Phase 4 Auto-Gen

This phased approach gives you the best return on investment.
