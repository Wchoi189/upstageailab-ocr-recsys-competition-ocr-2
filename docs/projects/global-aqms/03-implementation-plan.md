# Global AQMS Implementation Plan

**Date**: 2026-02-07
**Version**: 2.0 (Revised - Nuclear Refactor Approach)
**Status**: Draft - Ready for Execution

---

## Overview

This plan outlines the **nuclear refactor** approach to creating a global, stateless AQMS framework by consolidating three tools into a single CLI. No backward compatibility, migration, or legacy support.

## Goals

1. **Global Installation**: `pip install aqms-core` works anywhere
2. **Stateless Operation**: Framework code separate from project data
3. **Tool Consolidation**: Merge 3 CLIs into 1
4. **Clean Architecture**: Optimal design, no legacy constraints

---

## Phase 1: Foundation (Days 1-3)

### 1.1 Project Setup
- [ ] Create new `aqms-core` repository (or fork AgentQMS)
- [ ] Set up `pyproject.toml` with minimal dependencies
- [ ] Define package structure:
  ```
  aqms-core/
  ├── aqms/
  │   ├── core/          # Config, paths, utils
  │   ├── artifact/      # Artifact management
  │   ├── analyze/       # Code analysis (from ADT)
  │   ├── pulse/         # Pulse management (from Compass)
  │   ├── spec/          # Spec-kit (from Compass)
  │   ├── registry/      # Standards registry
  │   ├── cli.py         # Main CLI entry point
  │   └── mcp_server.py  # Unified MCP server
  └── pyproject.toml
  ```

### 1.2 Core Infrastructure
- [ ] Implement `ConfigLoader` with CWD-based project detection
- [ ] Implement `ProjectPaths` class (replaces old `paths.py`)
- [ ] Add `AQMS_PROJECT_ROOT` environment variable support
- [ ] Create `.aqms/` directory structure spec

**Deliverable**: Core infrastructure can detect/initialize projects

---

## Phase 2: CLI Surface (Days 4-7)

### 2.1 Command Groups

Consolidate 3 CLIs into unified structure:

| Group | Commands | Source |
|:------|:---------|:-------|
| **artifact** | create, validate, index | AgentQMS |
| **validate** | all, file, compliance | AgentQMS |
| **registry** | sync, resolve, suggest-header | AgentQMS |
| **analyze** | config, merges, hydra, deps, tree, complexity | agent-debug-toolkit |
| **ast** | search, query, lint, dump | agent-debug-toolkit |
| **pulse** | init, status, sync, export, checkpoint | project-compass |
| **spec** | constitution, specify, plan, tasks | project-compass |
| **init** | (scaffold new project) | New |
| **status** | (show project info) | New |

### 2.2 CLI Implementation
- [ ] Create `cli.py` with Click or Typer
- [ ] Implement each command group as subcommand
- [ ] Add unified help system
- [ ] Register `aqms` as console script in `pyproject.toml`

**Deliverable**: `aqms --help` shows all consolidated commands

---

## Phase 3: Tool Consolidation (Days 8-12)

### 3.1 Copy & Refactor Code

For each source tool:

#### From AgentQMS:
- [ ] Copy artifact management (`tools/core/artifacts/`)
- [ ] Copy validation system (`tools/compliance/`)
- [ ] Copy registry system (`.agentqms/registry.yaml` logic)
- [ ] **Refactor**: Remove legacy paths, simplify for CWD-based operation

#### From agent-debug-toolkit:
- [ ] Copy analyzers (`analyzers/` directory)
- [ ] Copy AST-grep wrapper (`astgrep.py`)
- [ ] Copy tree-sitter integration (`treesitter.py`)
- [ ] **Refactor**: Integrate into `aqms analyze` and `aqms ast` commands

#### From project-compass:
- [ ] Copy pulse manager (`src/core.py`)
- [ ] Copy spec-kit commands (CLI spec functions)
- [ ] Copy state schema (`src/state_schema.py`)
- [ ] **Refactor**: Integrate into `aqms pulse` and `aqms spec` commands

### 3.2 De-duplication
- [ ] Identify duplicate functionality (validation, file ops, etc.)
- [ ] Keep single implementation per feature
- [ ] Delete redundant code

### 3.3 Unused Tool Audit
- [ ] List all tools from 3 sources
- [ ] Mark rarely/never used tools
- [ ] **Exclude** unused tools from consolidation
- [ ] Document exclusions in `decisions/004-excluded-tools.md`

**Deliverable**: All commands functional under `aqms` CLI

---

## Phase 4: MCP Server (Days 13-15)

### 4.1 Unified MCP Server
- [ ] Create single `mcp_server.py`
- [ ] Expose all command groups as MCP tools
- [ ] Use dynamic project root detection
- [ ] Test multi-project isolation (run server separately per project)

### 4.2 MCP Tool Definitions
- [ ] Define schemas for each command
- [ ] Ensure all paths are absolute and CWD-aware
- [ ] Add project detection status to tool responses

**Deliverable**: Single MCP server exposing all AQMS features

---

## Phase 5: Packaging & Testing (Days 16-18)

### 5.1 Package Configuration
- [ ] Finalize `pyproject.toml` dependencies
- [ ] Bundle tree-sitter binaries (POSIX only)
- [ ] Create wheel for pip installation
- [ ] Test `pip install ./dist/aqms_core-2.0.0-py3-none-any.whl`

### 5.2 Integration Testing
- [ ] Test `aqms init` in empty directory
- [ ] Test multi-project scenarios (3+ projects side-by-side)
- [ ] Verify `AQMS_PROJECT_ROOT` override works
- [ ] Test all command groups end-to-end

### 5.3 Virtual Environment
- [ ] Auto-detect existing venv
- [ ] Offer to create venv if none exists
- [ ] Document venv workflow in README

**Deliverable**: Installable `aqms-core` package

---

## Phase 6: Documentation (Days 19-21)

### 6.1 User Documentation
- [ ] README with quickstart
- [ ] Command reference (all groups)
- [ ] Project structure guide (`.aqms/` layout)
- [ ] FAQ

### 6.2 Developer Documentation
- [ ] Architecture overview
- [ ] Contributing guide
- [ ] ADRs published in docs/

**Deliverable**: Complete documentation

---

## Phase 7: Release (Day 22)

### 7.1 Pre-Release Checklist
- [ ] All tests passing
- [ ] Documentation complete
- [ ] Version tagged as `v2.0.0`
- [ ] CHANGELOG prepared

### 7.2 Deprecate Legacy Tools
- [ ] Add deprecation notices to:
  - AgentQMS README
  - agent-debug-toolkit README
  - project-compass README
- [ ] Archive repositories (mark as deprecated)
- [ ] No migration tools provided

### 7.3 Publish
- [ ] Publish `aqms-core` to PyPI (or internal registry)
- [ ] Announce in project channels
- [ ] Update relevant documentation to reference `aqms-core`

**Deliverable**: `aqms-core v2.0.0` available globally

---

## Execution Strategy

### Parallel Workstreams

User can offload work to Claude Code while you handle planning/inspection:

| Day | Claude Code | You (Gemini) |
|:----|:------------|:-------------|
| 1-3 | Implement ConfigLoader, ProjectPaths | Review architecture, write specs |
| 4-7 | Build CLI structure, command groups | Test commands, refine UX |
| 8-12 | Copy/refactor tool code | Audit for unused tools, de-duplicate |
| 13-15 | Implement MCP server | Test MCP integration |
| 16-18 | Package and test | Review tests, fix bugs |
| 19-21 | Write user docs | Write ADRs, architecture docs |
| 22 | Final QA | Release checklist, announce |

### Rapid Iteration Loop

1. **Gemini** generates detailed spec for component
2. **Claude Code** implements per spec
3. **Gemini** inspects implementation, identifies issues
4. Repeat until complete

---

## Risk Mitigation

| Risk | Mitigation |
|:-----|:-----------|
| **Scope creep** | Stick to 7 command groups, defer extras to v2.1 |
| **Path bugs** | Write path resolution tests first |
| **Tree-sitter binaries** | Use pre-compiled wheels, document build process |
| **Multi-project conflicts** | Test `.aqms/` isolation thoroughly |

---

## Success Criteria

- [ ] `pip install aqms-core` works globally
- [ ] `aqms init` creates `.aqms/` structure
- [ ] All 7 command groups functional
- [ ] MCP server exposes all tools
- [ ] Documentation complete
- [ ] Legacy tools deprecated

---

## Next Steps

1. **Immediate**: Create `aqms-core` repository structure
2. **Day 1**: Implement `ConfigLoader` and `ProjectPaths`
3. **Day 2**: Create CLI skeleton with Click/Typer
4. **Begin parallel execution** with Claude Code

**Ready to start Phase 1?**
