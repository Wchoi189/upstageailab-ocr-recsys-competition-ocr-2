---
ads_version: "2.0"
id: SC-014
type: principle
tier: 1
priority: critical
memory_footprint: 80
auto_load: true
status: active
date: "2026-01-29"
---

# AgentQMS: AI-Native Architecture (Registry-Driven)

## Core Rules
- Registry is the single source of truth: use registry.yaml.
- One file = one function (specs | constraints | discovery | runtime).
- Tier intent: SST (laws), Framework (capabilities), Agents (identity), Workflows (execution).
- Any move/split requires registry sync + graph regen.

## Classification Shortcuts
- Framework = HOW it works (protocols, contracts, infra).
- Agents = WHO it is (identity/persona).
- Workflows = DO steps (runbooks, scripts).

## Required Actions
- Sync registry after standards changes: ./bin/aqms registry sync
- Regenerate mechanized graph: uv run python AgentQMS/tools/generate_mechanized_graph.py
- Load only required standards; avoid broad context.
