---
name: Qwen AgentQMS VS Code Tasks
overview: Add VS Code tasks that integrate Qwen Coder workflows for AgentQMS, providing quick access to Qwen-powered validation, artifact creation, and interactive tasks through the task runner.
todos:
  - id: add_qwen_validation_task
    content: "Add 'Qwen AgentQMS: Validate (Qwen)' task using ./.qwen/run.sh validate"
    status: pending
  - id: add_qwen_artifact_tasks
    content: Add Qwen artifact creation tasks for all types (plan, assessment, bug-report, design, research, template)
    status: pending
  - id: add_qwen_interactive_task
    content: "Add 'Qwen AgentQMS: Interactive' task with qwenPrompt input"
    status: pending
  - id: add_qwen_fix_task
    content: "Add 'Qwen AgentQMS: Apply Fixes' task combining validation and fixes"
    status: pending
  - id: add_qwen_prompt_input
    content: Add qwenPrompt input definition to inputs array
    status: pending
---

# Integrate Qwen AgentQMS into VS Code Tasks

## Overview
Add VS Code tasks to `.vscode/tasks.json` that leverage Qwen Coder for AgentQMS workflows, complementing existing Makefile-based tasks with Qwen-powered alternatives.

## Implementation Steps

### 1. Add Qwen Validation Task
- Task: "Qwen AgentQMS: Validate (Qwen)"
- Command: `./.qwen/run.sh validate`
- Purpose: Run Qwen-powered document validation agent

### 2. Add Qwen Artifact Creation Tasks
- Tasks for each artifact type using Qwen wrapper
- Command pattern: `./.qwen/run.sh create <type> ${input:artifactName} "${input:artifactTitle}"`
- Types: plan, assessment, bug-report, design, research, template

### 3. Add Qwen Interactive Task
- Task: "Qwen AgentQMS: Interactive"
- Command: `./.qwen/run.sh interactive "${input:qwenPrompt}"`
- Purpose: Run custom Qwen tasks with AgentQMS context
- New input: `qwenPrompt` for custom prompts

### 4. Add Qwen Direct CLI Task (Optional)
- Task: "Qwen AgentQMS: Direct CLI"
- Command: Direct `qwen` CLI invocation with AgentQMS context
- For advanced users who want full control

### 5. Add Qwen Fix Task
- Task: "Qwen AgentQMS: Apply Fixes"
- Command: Run validation, then use Qwen to apply fixes
- Combines validation + fix workflow

## Task Organization

Group Qwen tasks with clear naming:
- Prefix: "Qwen AgentQMS:"
- Distinguish from existing "AgentQMS:" tasks
- Use same presentation settings (dedicated panel, always reveal)

## Files to Modify

1. **`.vscode/tasks.json`**
   - Add new tasks array entries
   - Add `qwenPrompt` input definition

## Benefits

- Quick access to Qwen Coder from VS Code task runner
- No need to manually write prompts
- Consistent with existing AgentQMS task patterns
- Complements existing Makefile-based tasks