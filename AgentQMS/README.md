# AgentQMS: Quality Management System for AI Agents

**AI-Optimized Framework · Spec-Driven Standards · Context Bundle Discovery**

---

## Purpose

- **Context Bundle Discovery**: Auto-load relevant files based on task keywords
- **Spec-Driven Standards**: Machine-readable specifications organized by tier
- **Architecture Documentation**: Design principles and constraints for AI consumption
- **Validation Tools**: Compliance checking and artifact validation

---

## Directory Structure

```
AgentQMS/
├── specs/                    # Framework specifications
│   ├── tier1-contracts/      # Core contracts and interfaces
│   ├── tier2-framework/      # Framework-level standards (10 specs)
│   └── tier3-implementation/ # Implementation patterns
├── .agentqms/plugins/context_bundles/  # 17 discoverable bundles
├── tools/                    # QMS utilities (core, compliance, utils)
├── bin/aqms                  # CLI entry point
└── ARCHITECTURE.md           # Design principles (AI-optimized)
```

---

## Context Bundle System

### Constraints

| Tier | Token Limit | File Limit | Loading |
|:-----|:------------|:-----------|:--------|
| tier1 | <3000 | ≤6 | Auto-loaded |
| tier2 | No limit | No limit | On-demand |
| tier3 | No limit | No limit | Optional |

**Status** (2026-02-03): 4/17 bundles compliant

**Measurement**: `make qms-bundle-tokens`

### Available Bundles (17 total)

`ast-debugging-tools`, `hydra-configuration`, `ocr-debugging`, `ocr-experiment`, `v5-domains-standard`, and 12 more

**Discovery**: Bundles auto-load when AI agents mention trigger keywords

---

## CLI Tools

```bash
# Discovery
make qms-discover              # List all AgentQMS tools
make qms-status                # Framework status

# Context Bundles
make qms-bundle-tokens         # Measure bundle token usage
make qms-context TASK="..."    # Generate task-specific context

# Validation
make qms-validate              # Validate all artifacts
make qms-compliance            # Full compliance checks
make qms-boundary              # Verify framework boundaries

# Planning
make qms-plan NAME=my-plan     # Create implementation plan
make qms-plan-progress         # View/update plan progress

# Registry
make qms-registry              # Regenerate spec registry
```

**Binary**: `aqms validate --all`, `aqms context --task "..."`, `aqms monitor --report`

---

## Design Principles

From [`ARCHITECTURE.md`](ARCHITECTURE.md):

| Principle | Constraint | Result |
|:----------|:-----------|:-------|
| Consolidation | 300-600 tokens/spec | Single file per component |
| Shallow Hierarchy | Max 1 level nesting | 85% path overhead reduction |
| Token Discipline | Tables over prose | Max info density |
| Bundle Constraints | <3000 tokens tier1, ≤6 files | Efficient AI consumption |

---

## For AI Agents

**Quick Start**:
1. Read [`ARCHITECTURE.md`](ARCHITECTURE.md) for constraints
2. Browse `specs/tier2-framework/` for standards
3. Run `make qms-validate` before committing

**Common Tasks**:
- Hydra debugging → Load `hydra-configuration` bundle
- OCR pipeline → Load `ocr-debugging` or domain bundles
- Compliance → `make qms-compliance`
- Token measurement → `make qms-bundle-tokens`

---

## References

- [`ARCHITECTURE.md`](ARCHITECTURE.md) - Design principles
- [`.ai-instructions/`](../.ai-instructions/) - AI-optimized docs
- Tier1 specs - Artifact standards (naming, placement, workflow)

---

**Version**: ADS v2.0
**Last Updated**: 2026-02-03
