# Global AQMS Project Constitution

**Date**: 2026-02-07
**Project**: Global Stateless AQMS Framework
**Status**: Active

---

## Core Principles

### 1. Nuclear Refactor Philosophy
- **No backward compatibility** - Clean break from legacy AgentQMS
- **No migration tools** - Users must manually adopt new version
- **No deprecation period** - Immediate transition
- **No shims or fallback** - Simplicity over compatibility

> **Rationale**: Legacy support and migration complexity have corrupted past projects. Start fresh.

### 2. Stateless Architecture
- **Global installation**: Single `pip install aqms-core`
- **Project-local state**: All data in `.aqms/` directory per project
- **CWD-based detection**: No hardcoded paths relative to framework code
- **Environment override**: `AQMS_PROJECT_ROOT` for explicit control

### 3. Platform Constraints
- **POSIX paths only** - No Windows support (simplicity)
- **Python 3.10+** - Modern Python features
- **Virtual environment**: Auto-create/detect venv
- **Tree-sitter included** - Bundle pre-compiled binaries

### 4. CLI Design
- **Command name**: `aqms` (short, not `agentqms`)
- **Subcommand structure**: `aqms <group> <command>` (e.g., `aqms pulse init`)
- **Unified interface**: Single CLI consolidating 3 tools
- **MCP integration**: Single server exposing all features

### 5. Tool Consolidation
- **Merge**: agent-debug-toolkit → `aqms analyze`
- **Merge**: project-compass → `aqms pulse` + `aqms spec`
- **Preserve**: AgentQMS core → `aqms artifact`, `aqms validate`, `aqms registry`
- **Eliminate**: Unused tools flagged and removed

### 6. Development Workflow
- **Fast iteration**: Generate plans → Claude Code executes → Inspect
- **No testing debt**: Test as we build, not after
- **Document decisions**: ADRs for all major choices
- **State tracking**: Easy handover between sessions

---

## Constraints

| Category | Constraint | Enforcement |
|:---------|:-----------|:------------|
| **Complexity** | Avoid migration/fallback/shim logic | Code review |
| **Platform** | POSIX-only (Linux/macOS) | CI tests |
| **Dependencies** | Minimal external deps, bundle binaries | Dependency audit |
| **Documentation** | Single source of truth per topic | Doc review |

---

## Success Criteria

- [ ] Single `pip install aqms-core` works globally
- [ ] `aqms init` creates `.aqms/` in any directory
- [ ] All 3 legacy tools consolidated into `aqms` CLI
- [ ] No legacy code or compatibility layers
- [ ] All specs implemented and verified

---

## Anti-Patterns to Avoid

❌ **Avoid**: "We'll migrate users gradually..."
✅ **Do**: Clean break, users adapt or stay on old version

❌ **Avoid**: "Let's support both old and new paths..."
✅ **Do**: New path only, delete old code

❌ **Avoid**: "What if someone needs feature X?"
✅ **Do**: Build core features, extensions come later

---

**Reference**: [Original Design Document](file:///workspaces/docs/artifacts/design_documents/2026-02-07_1541_design_global-agentqms.md)
