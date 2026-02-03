---
spec_id: FW-CONTEXT-BUNDLES
spec_version: '1.0.0'
title: Context Bundle System
last_updated: '2026-02-03'
---

# Context Bundle System

## Purpose

Discovery indexes for auto-loading task-relevant files to AI agents.

## Constraints

| Tier | Token Limit | File Limit | Loading |
|:-----|:------------|:-----------|:--------|
| tier1 | <3000 | ≤6 | Auto-loaded |
| tier2 | No limit | No limit | On-demand |
| tier3 | No limit | No limit | Optional |

## Location

```
AgentQMS/.agentqms/plugins/context_bundles/*.yaml
```

## Bundle Structure

```yaml
name: bundle-name
title: Human-Readable Title
ads_version: '1.0'
scope: project
description: Brief purpose (1 line)
tags: [keyword1, keyword2]
triggers:
  keywords: [trigger1, trigger2]
tiers:
  tier1:
    name: Tier Name
    max_files: 6
    files:
    - path: relative/path/to/file
      priority: critical|high|medium|low
      mode: structure|reference|full
      description: Brief (≤5 words)
```

## Rules

**Descriptions**: ≤5 words, no verbose explanations
**Tags**: 3-5 max, focused keywords only
**Triggers**: 5-10 keywords, specific to task
**Tier1**: Critical files only, ≤6 files
**Tier2**: Advanced/specialized content
**Tier3**: Documentation, optional references

## Commands

```bash
make qms-bundle-tokens    # Measure token usage
make qms-audit-bundles    # Verify integrity
make qms-context TASK=... # Generate context
```

## Pitfalls

| Mistake | Fix |
|:--------|:----|
| Verbose descriptions | Max 5 words |
| Too many tags/triggers | 3-5 tags, 5-10 triggers |
| README in tier1 | Move to tier3 |
| >6 files in tier1 | Consolidate or tier2 |

## References

- [ARCHITECTURE.md](../../AgentQMS/ARCHITECTURE.md) - Bundle constraints
- [discovery.spec.md](../../AgentQMS/specs/tier2-framework/discovery.spec.md) - Tool catalog
