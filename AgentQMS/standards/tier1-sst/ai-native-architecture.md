---
ads_version: "1.0"
type: principle
tier: 1
priority: critical
memory_footprint: 80
auto_load: true
status: active
date: "2026-01-26"
---

# AgentQMS: AI-Native Architecture (Registry-Driven)

## Core Philosophy

**Documentation is Code for Agents.** The system must be **Registry-Driven**, **Pattern-Matched**, and **Categorically Pure**.

### 1. The Single Source of Truth: The Registry

The **`registry.yaml`** is the central router—not a state file, but the **discovery mechanism** that maps files to their governing standards.

- **Discovery Protocol**: Agents do not browse directories. They query `registry.yaml` to resolve the standards applicable to their current path or task.
- **Machine-First Schemas**: All logic, contracts, and compliance rules must be defined in YAML/JSON. Markdown is reserved for meta-theory and high-level onboarding (Tier 1).

### 2. Method-Locked Execution

- **Glob Association**: Every file modification must be preceded by a lookup in the `task_mappings`.
- **Context Injection**: Agents MUST load the specific YAML standards identified by the Registry before generating or refactoring code.

### 3. The 4-Tier Functional Hierarchy

To prevent categorical drift, every artifact in `AgentQMS/standards/` must align with one of these functional definitions:

| Tier | Name | Functional Definition | Example |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **SST** | **The Constitution.** Global "Laws" and meta-conventions that apply across the entire project. | `naming-conventions.yaml`, `file-placement.yaml` |
| **Tier 2** | **Framework** | **The Capabilities.** Technical specifications, API contracts, model configs, and logic components (AgentQMS core infrastructure). | `agent-infra/agent-architecture.yaml`, `coding/anti-patterns.yaml` |
| **Tier 3** | **Agents** | **The Personas.** Specific configuration for AI identities (Claude, Gemini, Researcher-Agent) - who they are, not how they work. | `claude/prompts.yaml`, `researcher/behavior.yaml` |
| **Tier 4** | **Workflows** | **The Operations.** Procedural sequences, step-by-step guides, and automated execution scripts. | `experiment-workflow.yaml`, `compliance-audit.py` |

### 4. Implementation Integrity

- **Registry Synchronization**: Any file relocation or creation requires an immediate, atomic update to `registry.yaml`.
- **Zero Ambiguity**: If a document contains both "Rules" and "Steps," it must be split into Tier 2 (Standards) and Tier 4 (Workflow).

## Critical Distinctions

### Framework vs Agents

The most common misclassification occurs between Tier 2 and Tier 3:

**Tier 2 (Framework)**: Infrastructure that defines **HOW the system works**
- Agent architecture patterns
- Communication protocols
- Model registries and platform configs
- Quality standards and anti-patterns

**Tier 3 (Agents)**: Persona configs that define **WHO the agent is**
- Identity and role definitions
- Persona-specific prompts
- Behavioral tuning parameters
- Agent-specific context preferences

### Classification Examples

| File | Tier | Reasoning |
| :--- | :--- | :--- |
| `agent-architecture.yaml` | **2 (Framework)** | System infrastructure - how agents are built |
| `claude-3-opus-config.yaml` | **3 (Agents)** | Persona identity - who this agent is |
| `ollama-models.yaml` | **2 (Framework)** | Platform registry - infrastructure catalog |
| `researcher-agent-prompts.yaml` | **3 (Agents)** | Persona behavior - how this agent thinks |
| `agent-feedback-protocol.yaml` | **2 (Framework)** | System protocol - how feedback works |
| `gemini-safety-settings.yaml` | **3 (Agents)** | Persona tuning - this agent's guardrails |

## Discovery Protocol

Agents MUST use `registry.yaml` for standard discovery:

```yaml
task_mappings:
  code_quality:
    description: "Code quality enforcement"
    standards:
      - tier2-framework/coding/anti-patterns.yaml
      - tier2-framework/coding/bloat-detection-rules.yaml
  
  agent_development:
    description: "Agent system development"
    standards:
      - tier2-framework/agent-infra/agent-architecture.yaml
      - tier2-framework/agent-infra/agent-feedback-protocol.yaml
```

## Tier 2 Subdirectory Structure

To maintain functional purity, `tier2-framework/` is organized by capability domain:

```
tier2-framework/
├── agent-infra/          # Agent system infrastructure
│   ├── agent-architecture.yaml
│   ├── agent-feedback-protocol.yaml
│   └── ollama-models.yaml
├── coding/               # Code quality standards
│   ├── anti-patterns.yaml
│   └── bloat-detection-rules.yaml
├── ocr/                  # OCR domain specifications
│   └── [domain files]
└── data/                 # Data contracts and schemas
    └── [data files]
```

## Implementation Rules

1. **Registry-Driven**: Query `registry.yaml` before applying any standard
2. **Glob Association**: Use file patterns for automatic standard application
3. **Context Injection**: Load only relevant standards to minimize memory
4. **Zero Ambiguity**: One canonical location per standard
5. **Functional Purity**: Tier assignment follows functional definition, not file type
6. **Infrastructure vs Identity**: "How it works" → Framework; "Who it is" → Agents

## Migration Directives

When refactoring existing standards:

1. **PRUNE**: Remove redundant tier categories (e.g., `tier3-governance`)
2. **CLASSIFY**: Apply functional definitions strictly (Infrastructure vs Identity)
3. **ORGANIZE**: Use subdirectories within Tier 2 for capability domains
4. **SYNCHRONIZE**: Update `registry.yaml` atomically with file moves
5. **VALIDATE**: Verify all cross-references and tier metadata fields
