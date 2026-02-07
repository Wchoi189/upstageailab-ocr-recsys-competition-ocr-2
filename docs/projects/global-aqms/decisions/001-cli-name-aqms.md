# ADR 001: CLI Name is `aqms`, Not `agentqms`

**Date**: 2026-02-07
**Status**: Accepted
**Decider**: User

---

## Context

Need to choose global CLI name for consolidated tool. Original assessment proposed `agentqms` to avoid confusion with legacy `aqms`.

## Decision

**Use `aqms` as the global CLI name.**

## Rationale

- `agentqms` is too long and difficult to type frequently
- `aqms` is already familiar to existing users
- No backward compatibility needed, so no confusion risk
- Brevity improves developer experience

## Consequences

**Positive:**
- Shorter commands: `aqms pulse init` vs `agentqms pulse init`
- Familiar to current AgentQMS users
- Less typing, faster workflows

**Negative:**
- Could cause brief confusion during transition if users have old `aqms` installed
- Mitigation: Document clean uninstall of legacy tools

## Implementation

- Package name: `aqms-core`
- Entry point: `aqms` (registered in `pyproject.toml` as console script)
- All documentation uses `aqms` consistently
