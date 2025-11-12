---
title: "Changelog Automation Process"
date: "2025-11-06"
type: "guide"
category: "ai_agent_process"
status: "active"
version: "1.0"
tags: ["changelog", "automation", "versioning"]
---

Changelog Automation Process
===========================

This guide covers semi-automation options for CHANGELOG.md generation, industry solutions, and a custom approach that integrates with your existing tracking system.

Industry Solutions
------------------

### 1. Conventional Commits + Semantic Release
**What**: Git commit message convention + automated tooling
**Tools**: `semantic-release`, `standard-version`, `conventional-changelog`
**How it works**:
- Commit messages follow format: `type(scope): description`
  - Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `chore`
  - Example: `feat(tracking): add experiment run tracking`
- Tool parses commits since last tag
- Generates changelog sections automatically
- Can auto-bump version and create git tags

**Pros**:
- Industry standard (used by Angular, Vue, etc.)
- Full automation possible
- Integrates with CI/CD
- Version bumping included

**Cons**:
- Requires discipline in commit messages
- Less flexible for structured entries (like your task-based format)
- May generate too many entries (spam risk)

**Example**:
```bash
npm install -g semantic-release
semantic-release --dry-run
```

### 2. Git-based Changelog Generators
**What**: Parse git log for patterns
**Tools**: `git-changelog`, `github-changelog-generator`, `clog-cli`
**How it works**:
- Scans git log for keywords: `fix:`, `feat:`, `BREAKING:`, etc.
- Groups by type and date
- Generates markdown

**Pros**:
- Works with existing commits
- No commit message discipline required upfront
- Can filter by date/author/branch

**Cons**:
- Less structured output
- Requires post-processing for your format
- May miss context (no task tracking integration)

### 3. Issue/PR-based Changelogs
**What**: Generate from GitHub/GitLab issues/PRs
**Tools**: `release-drafter`, `github-changelog-generator`
**How it works**:
- Scans closed issues/merged PRs since last release
- Uses labels/categories to group entries
- Generates changelog from issue titles/descriptions

**Pros**:
- Rich context from issue descriptions
- Can use labels for categorization
- Integrates with project management

**Cons**:
- Requires issue/PR discipline
- May not capture all changes
- Less useful if you don't use issues heavily

Semi-Automation Approach (Recommended)
---------------------------------------

### Hybrid: Generate Draft + Human Curate

**What we can automate**:
1. Collect changelog candidates from multiple sources:
   - Completed plans (from tracking DB)
   - Completed experiments (with summaries)
   - Git commits (conventional commits or keywords)
   - Deprecated docs (from version bump)
   - Tool catalog changes (new tools added)

2. Generate structured draft with sections:
   - Features (from completed plans)
   - Experiments (from completed experiments with summaries)
   - Fixes (from git commits with `fix:` prefix)
   - Documentation (from deprecated docs)
   - Tools (from catalog changes)

3. Pre-fill your format:
   - Date from `project_version.yaml`
   - Task structure from tracking DB
   - File references from git diff
   - Progress tracking from plan status

**What requires human decision**:
- Which entries to include (prevent spam)
- Summarization and grouping
- Priority/ordering
- Breaking changes identification
- Task completion status

### Implementation Strategy

**Phase 1: Data Collection Script**
- Query tracking DB for completed plans/experiments since last version
- Parse git log for conventional commits
- Check deprecated docs list
- Compare tool catalog (before/after)

**Phase 2: Draft Generator**
- Generate structured markdown draft
- Use templates matching your format
- Include metadata (files, dates, status)
- Output to `CHANGELOG.draft.md`

**Phase 3: Human Curation Workflow**
- Review draft
- Edit/remove entries
- Add summaries
- Merge into `CHANGELOG.md`

**Phase 4: Integration with Version Bump**
- Hook into `make version-bump`
- Auto-generate draft
- Prompt for review before commit

Recommended Implementation
--------------------------

### Option A: Lightweight (Git + Tracking DB)
**Automation level**: ~60%
- Parse git commits for `feat:`, `fix:`, `refactor:` patterns
- Query tracking DB for completed plans/experiments
- Generate draft with sections
- Human curates and merges

**Effort**: Low (1-2 hours)
**Maintenance**: Low

### Option B: Full Integration (Git + Tracking + Docs + Catalog)
**Automation level**: ~80%
- All of Option A, plus:
- Check deprecated docs from version bump
- Compare tool catalog snapshots
- Auto-detect breaking changes (git tags, version bumps)
- Generate structured draft with all metadata

**Effort**: Medium (4-6 hours)
**Maintenance**: Medium

### Option C: Conventional Commits + Semantic Release
**Automation level**: ~90%
- Adopt conventional commits standard
- Use `semantic-release` or `standard-version`
- Full automation with CI integration
- Requires commit message discipline

**Effort**: High (adoption + migration)
**Maintenance**: Low (once adopted)

Recommendation
--------------

**Start with Option A (Lightweight)**, then evolve to Option B if useful:

1. **Immediate value**: Leverages your existing tracking system
2. **Low risk**: Doesn't require commit message changes
3. **Fits your workflow**: Works with your task-based structure
4. **Easy to extend**: Can add more sources later

**Implementation**:
- Create `scripts/agent_tools/documentation/generate_changelog_draft.py`
- Integrate with `make version-bump` workflow
- Generate `CHANGELOG.draft.md` for review
- Human curates and merges

**Next Steps**:
1. Implement Option A (lightweight generator)
2. Test with next version bump
3. Iterate based on what's useful
4. Consider Option B if you want more automation

Would you like me to implement Option A (lightweight generator) that integrates with your tracking system and version bump workflow?
