---
ads_version: '2.0'
id: 'FW-037'
type: 'tool_catalog'
tier: 2
priority: 'high'
updated: '2026-02-03'
spec_version: '1.0.0'
description: 'Keyword mappings for context bundle task detection and routing'
dependencies:
  - FW-034
---

# Context Bundle Discovery Rules

> Keyword mappings for context bundle task detection and routing

## References

- [`ARCHITECTURE.md`](../../ARCHITECTURE.md) - Bundle constraints and tiering rules
- [`context-bundles.yaml`](../../../.ai-instructions/tier2-framework/context-bundles.yaml) - Bundle system spec
- Bundles: `AgentQMS/.agentqms/plugins/context_bundles/*.yaml`

## Specification

```yaml
agent: all
dependencies:
- FW-034
development:
- implement
- code
- develop
- feature
- function
- class
- module
- refactor
- rewrite
- build
- create
- add
- fix bug
- bug fix
documentation:
- document
- doc
- write docs
- readme
- guide
- manual
- tutorial
- update docs
- documentation
debugging:
- debug
- troubleshoot
- error
- fix
- broken
- issue
- problem
- crash
- exception
- traceback
- log
- investigate
planning:
- plan
- design
- architecture
- blueprint
- strategy
- assess
- evaluate
- analysis
- proposal
- roadmap
hydra-configuration:
- hydra
- omega
- conf
- config.yaml
- configuration
- hydra v5
- domains first
- package directive
- interpolation
hydra-v5-patterns:
- atomic architecture
- domain isolation
- flattening rule
- absolute interpolation
- self-mounting
- aliasing pattern
- namespace collision
- double wrap
- passive refactor cycle
ocr-architecture:
- OCRProjectOrchestrator
- domain separation
- ocr/pipelines
- ocr/domains
- detection domain
- recognition domain
- orchestration flow
- component interfaces
agent-configuration:
- agent
- multi-agent
- sub-agent
- ollama
- model
- ai assistant
- copilot
- claude
- gemini
- qwen
- side load
- agent system

```
