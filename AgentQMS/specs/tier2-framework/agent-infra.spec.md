---
ads_version: '2.0'
id: 'AG-006'
type: 'tool_catalog'
tier: 3
priority: 'high'
updated: '2026-02-03'
description: 'Ollama model catalog with specifications for Qwen model family'
dependencies:
  - SC-007
  - FW-034
---

# Ollama Models Configuration

> Ollama model catalog with specifications for Qwen model family

## Specification

```yaml
agent: qwen
dependencies:
- SC-007
- FW-034
endpoint: http://host.docker.internal:11434
inventory:
- name: qwen3-coder:30b
  role: Architect
  context_window: 1048576
  strengths:
  - Repo-scale refactoring
  - Complex logic
  - Multi-file impact analysis
  vram_requirement: 18GB
- name: qwen3:1.7b
  role: Utility / Janitor
  context_window: 32768
  strengths:
  - Log parsing
  - Metadata extraction
  - Schema validation
  vram_requirement: 1.4GB
- name: qwen3:4b-instruct
  role: Validator
  context_window: 262144
  strengths:
  - Thinking mode
  - Logical verification
  - Quality scoring
  vram_requirement: 2.5GB

```