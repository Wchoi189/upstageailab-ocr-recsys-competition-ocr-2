---
ads_version: "1.0"
title: "Context Bundling and Atomic Standards Pattern"
date: "2026-01-12 21:45 (KST)"
status: "active"
type: "design_document"
category: "architecture"
version: "1.0"
---

# Context Bundling and Atomic Standards Pattern

## Objective
Design a system that delivers **relevant** documentation to AI agents with **zero noise**.
Instead of loading massive monolithic standard files, we dynamically bundle small, "Atomic Standards" based on the user's current intent.

## The Pattern

### 1. Atomic Standards (The Data Layer)
**Principle**: "One Concern, One File".
Standards must be broken down into their smallest logical units. A file should only be loaded if it is directly relevant to the current task.

**Anti-Pattern (Monolith):**
`coding-standards.yaml` (Includes Testing, Async, Pydantic, ML, Logging, Exceptions...)
*   *Problem*: If I'm writing a simple Pydantic model, I don't need to read about "AsyncIO threadpools" or "Pytest fixtures". It wastes tokens and distracts the model.

**Pattern (Atomic):**
`standards/tier2-framework/coding/`
*   `python-core.yaml` (Generic Python rules)
*   `testing.yaml` (Pytest specific)
*   `async-concurrency.yaml` (Async/Await specific)
*   `pydantic-best-practices.yaml` (Validation specific)

### 2. Semantic Routing (The Control Layer)
Use a Router (Rule Set) to map **Tasks** to **File Paths**.
The router serves as the index for the Context Bundler.

**Structure (`standards-router.yaml`):**
```yaml
task_mappings:
  <task_name>:
    description: "Human readable description"
    triggers:
      keywords: ["list", "of", "trigger", "words"]
    standards:
      - path/to/atomic/standard_1.yaml
      - path/to/atomic/standard_2.yaml
```

### 3. Frontmatter Strategy (Lean Profile)
Atomic standards must use the **Lean Profile** to minimize overhead.
*   **Keep**: `type`, `tier`, `priority`, `agent`
*   **Remove**: `title`, `date`, `status`, `description` (unless critical)

## Implementation Guide

### Step 1: Identify Cohesion
Look at a standard file. Can different sections be used independently?
*   *Yes* -> **Split it.**
*   *No* -> Keep it together.

### Step 2: Extract & atomize
1.  Create a subdirectory for the domain (e.g., `coding/`, `infrastructure/`).
2.  Create separate YAML files for each concern.
3.  Add Lean Frontmatter to each.

### Step 3: Map in Router
Update `standards-router.yaml` to point to the new atomic files instead of the old monolith.

### Step 4: Validate
Run `compliance-checker.py` to ensure all new files adhere to the Lean Schema.
