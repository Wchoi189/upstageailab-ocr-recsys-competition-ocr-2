# [Compass:Lock] Core Directive 00: Meta-Cognition Protocol

## Purpose
Force deliberate reasoning before any file creation or state change.

## The [Compass:Reflection] Template

Before creating ANY artifact, output this reflection:

```
[Compass:Reflection]
- Type: (design|research|walkthrough|implementation_plan|bug_report|audit)
- Justification: Why is this file necessary for THIS pulse?
- Redundancy: Does an existing artifact cover this? Why not update it?
- Lifecycle: Transitional (delete after) or Archived (keep in history)?
```

## Behavioral Rules

1. **Challenge Vague Instructions**: If a milestone task is unclear, ASK for clarification. Do not guess.
2. **Evidence-Based Completion**: Cannot mark tasks complete without validation command output.
3. **Proactive Pruning**: If legacy code exists that should be cleaned, add to `active_blockers`.

## Anti-Patterns

| ❌ Banned                       | ✅ Required                                  |
| ------------------------------ | ------------------------------------------- |
| "I think..."                   | "The code shows..."                         |
| "Maybe we could..."            | "Will implement X by..."                    |
| Creating files outside staging | All artifacts in `pulse_staging/artifacts/` |
| Narrative summaries            | Indicator-only notes (max 20 words)         |
