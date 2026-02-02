# Core Infrastructure Specification

**Tier**: 2 (Framework)
**Scope**: Agent Architecture, Python Core, and Infrastructure layers.

## 1. Agent Architecture

**Role**: Semantic Glue between Tools and Standards.

### Architecture Layers
1.  **Identity Layer**: Who the agent is (Persona).
2.  **Context Layer**: What the agent knows (Context Bundles).
3.  **Tool Layer**: What the agent does (Execution).

### Feedback Protocol
*   **Loop**: `Plan` -> `Execute` -> `Verify` -> `Refine`.
*   **Failure**: If a tool fails 3 times, stop and ask user (or pivot strategy).

## 2. Python Core Standards

**Style Guide**:
*   **Type Hinting**: Mandatory for all function signatures.
*   **Docstrings**: Google Style. Required for all public methods.
*   **Imports**: Absolute imports only (`from AgentQMS.tools...`).

### prohibited-actions (Python)
*   **Wildcard Imports**: `from module import *` (BANNED).
*   **Mutable Defaults**: `def foo(l=[])` (BANNED).
*   **Print Statements**: Use `logging` or `rich.console`.

## 3. Multi-Agent Infrastructure
*   **Coordination**: Via Artifact handoff (tasks, plans).
*   **State**: Stateless execution preferred; state lives in file system artifacts.
*   **Ollama Models**: Use `qwen2.5-coder` for code, `llama3` for chat.
