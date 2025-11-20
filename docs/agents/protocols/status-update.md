# Status Update Protocol

## Purpose

This protocol defines when and how AI agents should provide periodic status updates to keep users informed of progress, blockers, and next steps.

## When to Provide Status Updates

**Required:**
- Every 5 major tasks completed
- When encountering blockers or errors
- On major milestones (phase completion, etc.)
- After 30+ minutes of continuous work

**Optional:**
- When switching between different types of work
- Before starting a large/complex task
- When discovering important information

## Status Update Format

### Template

```markdown
## Status Update: [Brief Title]

**Time:** [Current time/duration]
**Phase:** [Current phase or task area]

### ✅ Completed
- Task 1 description
- Task 2 description
- Task 3 description

### 🔄 In Progress
- Current task description
- Progress: [percentage or description]

### ⏭️  Next Steps
1. Next task 1
2. Next task 2
3. Next task 3

### 🚧 Blockers (if any)
- Blocker 1 description
- Blocker 2 description

### 📊 Overall Progress
- Tasks completed: X/Y
- Estimated completion: [timeframe or percentage]
```

### Example

```markdown
## Status Update: Documentation Reorganization - Phase 1 Complete

**Time:** 2025-11-20 14:30 KST
**Phase:** Phase 1 - Immediate Restructuring

### ✅ Completed
- Pruned .cursor/rules/prompts-artifacts-guidelines.mdc (167 → 60 lines)
- Enhanced AGENT_ENTRY.md with critical links and pre-commit checklist
- Created status-update.md protocol

### 🔄 In Progress
- Splitting docs/agents/system.md into core + references

### ⏭️  Next Steps
1. Extract operational commands to references/operations.md
2. Create quick-reference.md
3. Continue with Phase 2

### 📊 Overall Progress
- Phase 1: 3/4 tasks complete (75%)
- Overall: 3/12 tasks complete (25%)
```

## Guidelines

**Keep it concise:**
- Brief, scannable format
- Use bullet points
- Avoid redundant details

**Be specific:**
- Actual task names, not vague descriptions
- Clear next steps
- Quantifiable progress when possible

**Highlight blockers:**
- Clearly identify what's blocking progress
- Propose solutions if possible
- Don't hide problems

**Update progress trackers:**
- If working from an implementation plan, update its Progress Tracker
- Status updates complement, not replace, progress trackers

## Integration with State Tracking

When state tracking is available, use the session manager:

```python
from agent_qms.toolbelt import SessionManager, StateManager

session_mgr = SessionManager(StateManager())

# Track outcomes as you complete tasks
session_mgr.add_outcome("Completed Phase 1 restructuring")

# Track challenges/blockers
session_mgr.add_challenge("Need to refactor large system.md file")

# End session with summary
session_mgr.end_session(
    summary="Phase 1 documentation restructuring complete",
    outcomes=["3 tasks complete", "Files reduced in size"],
    challenges=["Large files need careful splitting"]
)
```

## Frequency

**Normal work:** Every 5 major tasks or 30-45 minutes

**Complex/long tasks:** More frequently (every 2-3 subtasks or 15-20 minutes)

**Simple tasks:** Less frequently (only at major milestones)

## Automation

For implementation plans with Progress Trackers:
- Update the tracker after each task
- Provide a status update every 5 tasks
- Tracker updates serve as micro-status updates
- Full status updates provide higher-level overview

## Benefits

- User knows work is progressing
- Early identification of blockers
- Clear visibility into next steps
- Easier to resume after interruptions
- Better context for future conversations
