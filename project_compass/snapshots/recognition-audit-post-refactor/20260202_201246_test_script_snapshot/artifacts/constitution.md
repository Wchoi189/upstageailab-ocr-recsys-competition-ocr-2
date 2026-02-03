# Project Constitution

## Principles
## Recognition Pipeline Audit - Guiding Principles

### Principle 1: Evidence-Based Analysis
- All findings must be backed by concrete evidence (code, configs, logs)
- No assumptions about what "should" exist without verification
- Document what IS, not what was intended

### Principle 2: Reference-Driven Comparison
- Detection pipeline is the V5 gold standard
- All structural patterns should align with detection unless documented otherwise
- Differences require explicit justification

### Principle 3: Surgical Scope Management
- Fix only V5 structural compliance issues
- Defer architectural redesign concerns
- Separate symptom from root cause

### Principle 4: Audit-Before-Action
- Complete comprehensive audit before proposing fixes
- Identify ALL missing pieces, not just first error
- Map dependencies between components

### Principle 5: Tool-Assisted Investigation
- Leverage existing audit scripts (master_audit.py, hydra_target_linter.py)
- Use ADT meta tools for structural analysis
- Automate repetitive checks

### Principle 6: Incremental Validation
- Each finding should be independently verifiable
- Proposed fixes must be testable in isolation
- Maintain fast_dev_run as validation gate

### Principle 7: Knowledge Preservation
- Document WHY things are the way they are
- Reference historical decisions (refactor tracking, DEBUG sessions)
- Create artifacts for future troubleshooting

## Established
Date: 2026-02-02T19:43:49.847892
Tool: Project Compass v2
