# ADR 002: Nuclear Refactor - No Backward Compatibility

**Date**: 2026-02-07
**Status**: Accepted
**Decider**: User

---

## Context

Original assessment proposed conservative approach with migration tools, backward compatibility, and deprecation periods. User feedback rejected this complexity.

## Decision

**Nuclear refactor: Clean break with zero backward compatibility.**

## Rationale

> "Backward Compatibility is not required. This has caused more problems than has solved. Improvements and upgrades are desired."
>
> "Legacy and shims have corrupted the project in the past. A clean break is desired."

Key points:
1. **Past experience**: Legacy support caused project corruption
2. **Simplicity**: Migration/deprecation logic adds complexity
3. **User preference**: Immediate adoption or stay on old version

## Consequences

**Positive:**
- **Simpler codebase**: No compatibility layers, shims, or fallback logic
- **Faster development**: No need to test old + new paths
- **Cleaner architecture**: Design for future, not constrained by past

**Negative:**
- **User migration burden**: Manual work required to adopt new version
- **Potential resistance**: Some users may not upgrade
- Mitigation: Clear documentation, compelling features

## What This Means

### ✅ We Will:
- Create entirely new codebase (can copy/fork AgentQMS)
- Design optimal architecture without constraints
- Delete legacy code paths completely
- Document clean installation process

### ❌ We Will NOT:
- Provide migration scripts or tools
- Support old command syntax
- Maintain deprecated features
- Offer compatibility mode

## Implementation

1. Create new `aqms-core` package
2. Copy relevant code from AgentQMS/project-compass/agent-debug-toolkit
3. Refactor without preserving old interfaces
4. Document as "v2.0" - incompatible with v1.x
5. Archive legacy repositories with clear "deprecated" notice
