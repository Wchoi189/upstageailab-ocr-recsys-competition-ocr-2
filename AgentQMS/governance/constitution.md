# AgentQMS Constitution

**Governance**: Supreme Law
**Date**: 2026-02-02

## Preamble
AgentQMS is an AI-Native Quality Management System designed to be managed **by agents, for agents**.

## Core Principles

### 1. The Registry is Truth
Changes to the system must be reflected in the Registry (`specs/`) first. Code follows specs.

### 2. Atomic Modularity
*   **Tiny Specs**: Files must remain < 600 tokens.
*   **Tiny Components**: Functions must be < 50 lines.

### 3. Explicit Context
Agents must load **only** what they need. No "global context" dumping.

### 4. Continuous Validation
Every action (`create`, `edit`) must be validated against `tier1-contracts`.

## Migration Status
*   **Legacy**: `standards/` (Archived).
*   **Current**: `specs/` (Active).
