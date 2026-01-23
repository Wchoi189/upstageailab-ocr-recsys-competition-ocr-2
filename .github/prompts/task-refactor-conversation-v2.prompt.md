---
name: Task Refactor Conversation to Implementation Plan v2
description: Convert a conversation thread into a structured implementation plan for coding agents.
---

# AI Agent Refactoring Instruction Template

## Context
Project: {PROJECT_NAME}
Current State: {BRIEF_STATE_DESCRIPTION}
Objective: Convert conversation thread into executable implementation plan

## Instructions for AI Agent

### 1. Analysis Phase
Parse the attached conversation and extract:

- **Core Problems:** List technical issues requiring resolution
- **Proposed Solutions:** Enumerate all suggested implementations
- **Dependencies:** Identify external packages, APIs, infrastructure
- **File Operations:** Categorize as CREATE, MODIFY, DELETE, ARCHIVE

### 2. Plan Structure Requirements

#### Output Format
```yaml
project: {PROJECT_NAME}
sessions:
  - id: session_01
    focus: {CONCISE_FOCUS_AREA}
    duration_estimate: {MINUTES}
    cost_estimate: {USD}

    files:
      create:
        - path: {RELATIVE_PATH}
          purpose: {ONE_LINE_PURPOSE}
          dependencies: [{PKG1}, {PKG2}]

      modify:
        - path: {RELATIVE_PATH}
          changes: [{CHANGE_TYPE}]

      archive:
        - path: {RELATIVE_PATH}
          reason: {DEPRECATION_REASON}

    validation:
      - type: {TEST_TYPE}
        command: {VALIDATION_CMD}
```

#### Session Splitting Rules
- Max duration: 120 minutes per session
- Max files: 8 files modified per session
- Dependencies: Group related changes together
- Sequence: Order by dependency chain (infrastructure → core → features)

### 3. Optimization Constraints

#### Token Budget:
- File paths: Use relative paths only
- Descriptions: Max 10 words per item
- Code snippets: Exclude from plan (reference only)
- Comments: Prohibited in YAML output

#### Standardization:
- Use consistent naming: {component}_{action}.py
- Group by directory structure
- Alphabetize within groups

### 4. Output Sections

#### A. Executive Summary (Max 50 tokens)
- Problem: {ONE_SENTENCE}
- Solution: {ONE_SENTENCE}
- Impact: {METRIC} reduction in {RESOURCE}

#### B. File Manifest
- CREATE: {COUNT} files
- MODIFY: {COUNT} files
- ARCHIVE: {COUNT} files
- DELETE: {COUNT} files

#### C. Session Breakdown
For each session:

```
Session {N}: {FOCUS}
├── Prerequisites: [{ITEMS}]
├── Deliverables: [{ITEMS}]
├── Validation: {TEST_COMMAND}
└── Handoff: {NEXT_SESSION_INPUT}
```

#### D. Resource Requirements
- External Packages: [{PKG: VERSION}]
- Infrastructure: [{SERVICE: CONFIG}]
- API Keys: [{SERVICE: SCOPE}]
- Hardware: [{REQUIREMENT: SPEC}]

### 5. Validation Rules
Before outputting plan:

- All file paths exist or are valid new paths
- No circular dependencies between sessions
- Each session has clear entry/exit criteria
- Cost estimates include API + infrastructure
- Handoff protocol defined for context saturation

## Attached Conversation
{PASTE_CONVERSATION_HERE}

## Expected Output
Provide the implementation plan in the YAML format specified above. Exclude:

- Verbose explanations
- Code implementations
- Tutorial content
- Historical context

Include only:

- Actionable file operations
- Dependency chains
- Validation commands
- Resource requirements

**Optimization Target:** Plan must be executable by autonomous agents with zero human interpretation required.

## Example Usage
```yaml
# For the AgentQMS conversation:
PROJECT_NAME: AgentQMS
BRIEF_STATE_DESCRIPTION: "Transition from config-heavy prototype to Actor-Model architecture with 85% token reduction achieved, addressing state persistence crisis"

# Agent processes conversation and outputs:
# - 3 sessions (Cleanup, IACP, AutoGen)
# - 12 file operations
# - 4 external dependencies
# - Validation commands per session
```
