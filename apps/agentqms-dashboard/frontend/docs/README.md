---
title: "Dashboard Documentation Index"
type: meta
status: active
created: 2025-12-08 14:30 (KST)
updated: 2025-12-08 14:30 (KST)
phase: 1-2
priority: high
tags: [index, navigation, toc]
---

# AgentQMS Manager Dashboard Documentation

**Status**: Phase 1-2 complete. Phase 3 (backend bridge) pending.

## Quick Links

**START HERE**: [Progress Tracker](meta/2025-12-08-1430_meta-progress-tracker.md)

**By Category**:
- **Architecture** — [Frontend Patterns](architecture/2025-12-08-1430_arch-frontend-patterns.md), [Diagrams](architecture/2025-12-08-1430_arch-system-diagrams.md)
- **API** — [Contracts](api/2025-12-08-1430_api-contracts-spec.md), [Principles](api/2025-12-08-1430_api-design-principles.md)
- **Development** — [Bridge Guide](development/2025-12-08-1430_dev-bridge-implementation.md), [Features](development/2025-12-08-1430_dev-dashboard-features.md)
- **Plans** — [Roadmap](plans/in-progress/2025-12-08-1430_plan-development-roadmap.md), [Risk](plans/notes/2025-12-08-1430_plan-risk-assessment.md)
- **Meta** — [AI Instructions](meta/2025-12-08-1430_meta-ai-instructions.md), [Session Handovers](meta/)

## Status

| Phase | Result | Timeline | Notes |
|-------|--------|----------|-------|
| 1 | ✅ Complete | 24h | Documentation & architecture |
| 2 | ⚠️ Incomplete | 1h actual | Backend bridge not implemented |
| 3 | 🔴 Pending | 4-6 weeks | Implementation awaiting start |

## Blockers

1. 🔴 **Missing Backend Bridge** — `AgentQMS/agent_tools/bridge/` (20-30h to implement)
2. 🔴 **No Integration Tests** — Python ↔ React tests not written (15h)
3. 🔴 **Repo Status Unknown** — GitHub dashboard repo needs sanity check (2h)

## Workflow

**New Session?** Read [Progress Tracker](meta/2025-12-08-1430_meta-progress-tracker.md) → [Latest Handover](plans/session/2025-12-08-1300_session-handover-phase2-complete.md) → [AI Instructions](meta/2025-12-08-1430_meta-ai-instructions.md)

**Continuing?** Update Progress Tracker weekly. Follow naming convention: `YYYY-MM-DD-HHMM_[category]-[descriptor].md`. Add frontmatter to all docs.

**Ending Session?** Create session handover with continuation prompt. Update Progress Tracker.

## Directory Structure

```
├── architecture/     # System design
├── api/             # API specs
├── development/     # Implementation guides
├── plans/           # Roadmaps, notes, sessions
│   ├── draft/
│   ├── in-progress/
│   ├── complete/
│   ├── notes/
│   └── session/     # Session handovers
├── meta/            # Progress tracker, AI protocol
└── README.md        # This file
```

## Session Handovers

- [Phase 1: Complete](plans/session/2025-12-08-1700_session-handover-phase1.md) (2025-12-08 17:00)
- [Phase 2: Start](plans/session/2025-12-08-1200_session-handover-phase2-start.md) (2025-12-08 12:00)
- [Phase 2: Complete (Context Saturation)](plans/session/2025-12-08-1300_session-handover-phase2-complete.md) (2025-12-08 13:00)
