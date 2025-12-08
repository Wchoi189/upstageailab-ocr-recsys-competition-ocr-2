---
title: "AI Documentation & Development Protocol"
type: meta
status: active
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 14:30 (KST)
phase: 1-2
priority: high
tags: [meta, ai-protocol, documentation, guidelines, workflow]
---

# AI Documentation & Development Protocol

## Core Rules

1. **Documentation First** - Plan before code, track changes, maintain progress tracker
2. **Hierarchical Structure** - Organized directories with consistent naming
3. **Frontmatter Required** - Every markdown file must have YAML header
4. **Session Handovers** - Preserve context at end of each session

## Directory Structure

```
docs/agentqms-manager-dashboard/
├── architecture/      # System design
├── api/               # API specs
├── development/       # Implementation guides
├── plans/
│   ├── draft/        # Early-stage planning
│   ├── in-progress/  # Active roadmaps
│   ├── complete/     # Finished (archive)
│   └── notes/        # Research, risk analysis
├── meta/             # Session handovers, progress, protocol
└── README.md         # Navigation hub
```

## File Naming

**Pattern**: `YYYY-MM-DD-HHMM_[category]-[descriptor].md`

**Categories**: `arch-`, `api-`, `dev-`, `plan-`, `meta-`

**Timestamp**: `YYYY-MM-DD-HHMM` (KST)

## Frontmatter (MANDATORY)

```yaml
---
title: "Title"
type: [architecture|api|development|plan|meta]
status: [draft|in-progress|complete|archived|active]
created: YYYY-MM-DD HH:MM (KST)
updated: YYYY-MM-DD HH:MM (KST)
phase: [1|2|3|bridge]
priority: [critical|high|medium|low]
tags: [keywords]
---
```

Update `updated:` field when editing.

## Workflow

**Before**: Read progress tracker (`meta/2025-12-08-1430_meta-progress-tracker.md`)

**During**: Create docs with proper naming/frontmatter. Update progress tracker weekly. Use feature branch: `feature/agentqms-dashboard-integration`

**After**: Update frontmatter (status, updated timestamp). Update progress tracker. Create session handover in `plans/session/` (see format below)

## Session Handover

**When**: End of session or major milestone

**Filename**: `plans/session/YYYY-MM-DD-HHMM_meta-session-handover-[phase]-[descriptor].md`

**Required Sections**:
- Session Summary (what was done)
- Current Status (where we are)
- Completed Items (checklist)
- Blocked Items (what's stuck, why)
- Pending Work (next tasks)
- Continuation Prompt (instructions for next session)

**Continuation Prompt Must Include**:
- Context (previous session, phase, blockers)
- Immediate Next Task (numbered steps)
- Success Criteria (measurable outcomes)
- Key Doc References

## Documentation Maintenance

### Weekly Audit Checklist
- [ ] Progress tracker updated (status, timeline, blockers)
- [ ] All new documents have frontmatter
- [ ] File timestamps are current
- [ ] Broken cross-references fixed
- [ ] Tags are accurate for search
- [ ] Archived docs moved to appropriate status

### Monthly Review Checklist
- [ ] Directory structure still makes sense
- [ ] Documentation drift assessed (docs vs. code)
- [ ] Obsolete documents archived or deleted
- [ ] Phase completion status accurate
- [ ] Dependencies and blockers still valid
- [ ] Risk assessment updated

### Phase Completion Checklist
- [ ] All deliverables documented
- [ ] Lessons learned captured
- [ ] Technical debt tracked
- [ ] Next phase context preserved
- [ ] Session handover created
- [ ] Progress tracker finalized

---

## Common Tasks & How to Do Them

### Task: Add a New Architectural Document

1. **Create file**: `docs/agentqms-manager-dashboard/architecture/2025-12-08-1430_arch-[descriptor].md`
2. **Add frontmatter**:
   ```yaml
   ---
   title: "[Human Title]"
   type: architecture
   status: draft
   created: 2025-12-08 14:30 (KST)
   updated: 2025-12-08 14:30 (KST)
   phase: [which phase]
   priority: [high|medium|low]
   tags: [keywords]
   ---
   ```
3. **Write content**: Use clear headers, examples, diagrams
4. **Add to README.md**: Update TOC with new document
5. **Link from progress tracker**: If relevant to current phase
6. **Set status to in-progress or complete** when done

### Task: Update Progress Tracker

1. **Open**: `meta/2025-12-08-1430_meta-progress-tracker.md`
2. **Update**:
   - Change `updated:` timestamp to current time
   - Mark completed items with ✅
   - Add new blockers if identified
   - Adjust time estimates if needed
3. **Add note**: Brief comment on what changed
4. **Commit**: `git add`, `git commit -m "Update progress tracker: [brief note]"`

### Task: Create a Session Handover

1. **Create file**: `plans/session/YYYY-MM-DD-HHMM_meta-session-handover-phase[N]-[descriptor].md`
2. **Fill template**:
   - Summary of session work
   - Completed items (✅)
   - Blocked items (🔴)
   - Pending work
   - Key decisions
   - File references
   - Continuation prompt
3. **Save and commit**: Push to feature branch
4. **Notify team**: Share handover summary

### Task: Fix Broken Cross-References

1. **Search**: `grep -r "path/to/old/file" docs/agentqms-manager-dashboard/`
2. **Update**: Replace with new path
3. **Verify**: Check that new path exists
4. **Update frontmatter**: Change `updated:` timestamp
5. **Commit**: `git add`, `git commit -m "Fix cross-reference: [from] → [to]"`

---

## Preventing Common Mistakes

### ❌ DON'T:
- Manually move files without updating cross-references
- Use old naming convention (e.g., `DATA_CONTRACTS.md`)
- Skip frontmatter on new documents
- Forget to update timestamps
- Store loose files in root directory
- Create new subdirectories without updating README.md
- Commit directly to `main` (use feature branch)
- Leave work undocumented (always write it down)

### ✅ DO:
- Update progress tracker weekly
- Use standardized naming convention
- Include complete frontmatter
- Update timestamps when editing
- Organize files by type (architecture/api/dev/plan/meta)
- Cross-reference liberally for discoverability
- Create feature branches for all work
- Document decisions and blockers
- Review README.md TOC regularly

---

## Maintenance

**Weekly**: Update progress tracker. Check frontmatter on new docs. Fix broken references.

**Monthly**: Review directory structure, assess doc drift, validate blocker status.

**Phase End**: Document deliverables, capture lessons, track technical debt.
