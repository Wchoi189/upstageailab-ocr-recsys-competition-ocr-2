# Global AQMS Project Documentation

This directory contains all planning, design, and decision documents for the **Global Stateless AQMS Framework** project.

## 📁 Structure

```
global-aqms/
├── 00-constitution.md       # Core principles & constraints (READ FIRST)
├── 01-requirements.md       # Requirements specification (TODO)
├── 02-architecture.md       # Technical architecture (TODO)
├── 03-implementation-plan.md # 22-day phased execution plan
├── 04-task-breakdown.md     # Granular task checklist (TODO)
│
├── decisions/               # Architecture Decision Records (ADRs)
│   ├── 001-cli-name-aqms.md
│   ├── 002-nuclear-refactor-no-compat.md
│   ├── 003-posix-only-platform.md
│   └── ... (more ADRs as decisions are made)
│
├── specs/                   # Component specifications
│   └── (TBD - e.g., config-loader.spec.md)
│
└── state/                   # Session state tracking
    ├── current-phase.md     # What phase we're in
    ├── blockers.md          # Current blockers (TODO)
    └── handover.md          # Session handover notes (TODO)
```

## 🚀 Quick Start

### For New Sessions:

1. **Read**: [00-constitution.md](file:///workspaces/docs/projects/global-aqms/00-constitution.md) - Understand core principles
2. **Check**: [state/current-phase.md](file:///workspaces/docs/projects/global-aqms/state/current-phase.md) - See current status
3. **Review**: [decisions/](file:///workspaces/docs/projects/global-aqms/decisions/) - Understand why decisions were made
4. **Execute**: [03-implementation-plan.md](file:///workspaces/docs/projects/global-aqms/03-implementation-plan.md) - Follow the plan

### For Making Progress:

1. Pick task from implementation plan
2. Create spec in `specs/` if needed
3. Implement with Claude Code
4. Update `state/current-phase.md`
5. Document decisions in `decisions/` if adding ADRs

## 📋 Document Types

| Type | Purpose | When to Create |
|:-----|:--------|:---------------|
| **Constitution** | Immutable principles | Once at project start |
| **Requirements** | What we need to build | After design review |
| **Architecture** | How we'll build it | After requirements locked |
| **Implementation Plan** | When and by whom | After architecture approved |
| **ADRs** | Record major decisions | Whenever significant choice made |
| **Specs** | Detailed component design | Before implementing complex features |
| **State** | Track progress | After each work session |

## 🎯 Key Decisions

All major decisions are documented as ADRs in `decisions/`. Current decisions:

- **ADR 001**: CLI name is `aqms` (not `agentqms`)
- **ADR 002**: Nuclear refactor, no backward compatibility
- **ADR 003**: POSIX-only platform support

## 📊 Current Status

**Phase**: Pre-Implementation
**Status**: Planning complete, awaiting approval to start Phase 1
**Next**: Implement ConfigLoader and ProjectPaths

See [state/current-phase.md](file:///workspaces/docs/projects/global-aqms/state/current-phase.md) for details.

---

**Project Goal**: Create a globally installable, stateless AQMS framework that consolidates 3 separate tools into a single CLI with unified MCP server.
