---
title: "AgentQMS Framework Maintenance & Evolution Design"
date: "2025-11-24 00:00 (KST)"
audience: maintainer
status: draft
version: "0.1"
tags: ["architecture", "maintenance", "roadmap"]
---

## Purpose

- Define how the AgentQMS framework should evolve while preserving its containerized design (only `.agentqms/` and `AgentQMS/` at project root).
- Provide maintainers with a concise reference for planning structural changes, documentation integration, and future capabilities.
- Align implementation tools, conventions, and knowledge layout around AI-agent-first usage.

## Current High-Level Architecture

- **Runtime / State**: `.agentqms/`
  - `settings.yaml` – project configuration (pre-resolution).
  - `effective.yaml` – merged runtime view of framework + project config.
  - `state/` – derived indices, caches, and (proposed) `architecture.yaml`.
- **Framework Container**: `AgentQMS/`
  - `agent_tools/` – implementation tools (artifact_workflow, validation, compliance, utilities).
  - `conventions/` – artifact types, schemas, templates, QMF manifest.
  - `knowledge/` – framework knowledge surface (agent protocols, references, templates, meta).

Key principle: **All framework behavior and knowledge must be discoverable from `.agentqms/effective.yaml` and `AgentQMS/` only.**

## Documentation & Knowledge Design

- **Goal**: Make all instructions AI-agent oriented, with human narrative strictly optional.

### Knowledge Layout

- `AgentQMS/knowledge/agent/`
  - Ultra-concise instructions and protocols for AI agents.
  - Example: `system.md` (single source of truth), CLI quick references, tool catalogs.
- `AgentQMS/knowledge/protocols/`
  - Governance / development / testing protocols expressed as compact rule sets.
- `AgentQMS/knowledge/references/`
  - Technical / architectural references needed when agents or maintainers debug deeper issues.
- `AgentQMS/knowledge/templates/`
  - Templates for artifacts and documentation (minimal prose, mostly structure).
- `AgentQMS/knowledge/meta/`
  - Maintainer-facing docs, export notes, and design discussions (including this file).

### Style Constraints for Agent-Facing Docs

- **Instructions, not tutorials**:
  - Prefer bullets and checklists; avoid long paragraphs.
  - One concept per bullet, 1–3 lines maximum.
- **Command-first**:
  - Show the exact command or pattern to follow, not a story.
- **No duplicated rules**:
  - Rules live in one canonical protocol or SST file and are only cross-referenced elsewhere.

## Artifact & Schema System

### Conventions Layer

- `AgentQMS/conventions/q-manifest.yaml`
  - Declares **artifact_types** with:
    - `name` – logical type (e.g., `implementation_plan`, `assessment`, `bug_report`).
    - `template` – path to the markdown template.
    - `schema` – path to the JSON schema under `schemas/`.
    - `location` – base directory for storing artifacts.
- `AgentQMS/conventions/schemas/`
  - JSON Schemas for each artifact type.
- `AgentQMS/conventions/templates/`
  - Minimal templates aligned with schemas and naming/frontmatter rules.

### New Bug Report Schema (Summary)

- **Schema file**: `AgentQMS/conventions/schemas/bug_report.json`
- **Manifest entry**: `artifact_types[].name == "bug_report"` in `q-manifest.yaml`.
- **Key fields** (required):
  - `title`, `author`, `date`, `status`, `severity`, `summary`.
- **Additional fields**:
  - `tags`, `steps_to_reproduce`, `expected_behavior`, `actual_behavior`,
    `root_cause`, `fix_summary`, `testing_notes`.

This schema is designed to:

- Align with the Bug Fix Protocol (root cause, fix, testing).
- Stay simple enough for automated generation and validation.

## Planned Architectural State Representation

- **File**: `.agentqms/state/architecture.yaml`
- **Role**:
  - Compact, machine-readable summary of:
    - Component layout (`agent_tools`, `conventions`, `knowledge`).
    - Capability map (e.g., `plan_generation`, `quality_assessment`).
    - Pointers to key docs (`agent_sst`, protocol roots, references).
    - Status of experimental features (e.g., smart context loading).
- **Usage**:
  - Agents and tools can query this file instead of scanning the full repo.
  - Future automation can regenerate it from `effective.yaml` and reference docs.
  - For now this file is maintained manually; automation is deferred until the design is validated.

## Smart Context Loading – Future Capability

- **Current status**: Documented as a reference (`knowledge/references/context_optimization/smart-context-loading.md`),
  not an active protocol.
- **Target state**:
  - An auto-activated protocol in `knowledge/agent/` that:
    - Chooses context bundles per task type.
    - Enforces progressive context loading and caching.
  - Backed by a slimmed-down reference in `knowledge/references/`.
- **Constraints**:
  - Must be thoroughly tested before becoming binding for agents.
  - Should expose a small, stable API or configuration surface in `.agentqms/state/architecture.yaml`.

## Roadmap for Future Revisions

### Phase 1 – Consolidation & Integration

- Move relevant docs from `docs/` into `AgentQMS/knowledge/` following the target layout.
- Prune verbose, human-oriented content into `knowledge/meta` or archives.
- Normalize frontmatter (`audience`, `status`, `domain`) across protocols and references.
- Ensure `q-manifest.yaml` and `schemas/` cover all active artifact types (including `bug_report`).

### Phase 2 – Agent-First Protocol Refinement

- Rewrite governance/development/testing protocols into strict agent-oriented format.
- Eliminate overlapping rules by:
  - Centralizing artifact rules in a single protocol.
  - Creating a single import/loader protocol plus one reference.
- Prefer **lightweight, human-triggered enforcement** for brevity:
  - Embed brevity reminders into artifact-generation and wrapper scripts.
  - Optionally provide soft-check utilities for maintainers instead of hard linting.

### Phase 3 – Architecture State & Tooling

- Implement generator for `.agentqms/state/architecture.yaml`:
  - Input: `effective.yaml`, `q-manifest.yaml`, key reference docs.
  - Output: compact capability and component map.
- Integrate architecture state into:
  - Tool discovery.
  - Agent context pre-loading (e.g., “give me all rules for capability X”).

### Phase 4 – Smart Context Loading

- Stabilize the design in `smart-context-loading.md` and extract:
  - A minimal, enforceable protocol for agents.
  - A simple configuration schema for context bundles.
- Add telemetry/metrics hooks (optional) to measure:
  - Context access time.
  - Task completion accuracy before/after activation.

## Maintainer Decisions

- **Framework history near the framework**: Only this design document is kept with the framework container; other project history (artifacts, audits, protocol assessments) should live outside `AgentQMS/`.
- **Architecture state management**: `.agentqms/state/architecture.yaml` will be maintained manually for now; any generator or automation will be added only after additional research.
- **Agent-doc brevity enforcement**: No strict automated linting; brevity is enforced primarily via reminders embedded in artifact-generation/wrapper scripts and occasional human review.

## Open Questions for Maintainers

- Which external tools (IDEs, CI, monitoring) should integrate with the architecture state and capability map?

Maintainers can use this document as a living plan: update phases, capture decisions, and record tradeoffs as the framework evolves.