# AgentQMS Architecture

**Version**: 1.0 (Phase 7.2 - Spec-Kit Migration)
**Last Updated**: 2026-02-03

## Core Principles

| Principle | Constraint | Target |
|:----------|:-----------|:-------|
| **Consolidation** | Single file per component | 300-600 tokens/spec |
| **Shallow Hierarchy** | Max 1 level nesting under tiers | Flat structure |
| **Token Discipline** | Tables over prose, cross-links over duplication | ≤3000 tokens/tier1 |
| **Bundle Constraints** | Discovery indexes, not documentation dumps | ≤6 files/bundle |
| **Spec Versioning** | Semantic versioning in frontmatter | v1.0.0+ |

### Bundle Constraints

| Tier | Purpose | Token Limit | File Limit | Loading |
|:-----|:--------|:------------|:-----------|:--------|
| tier1 | Critical references | <3000 | ≤6 | Auto-loaded |
| tier2 | Patterns, debugging | No limit | No limit | On-demand |
| tier3 | Examples, tests, manuals | No limit | No limit | Optional |

**Golden Rule**: Reference guides instead of embedding them

### Tiering Best Practices

**Metrics**:
- **Tier1 tokens** (auto-load overhead) - PRIMARY METRIC
- Total tokens - secondary

**Content Rules**:
- **Tier1**: Critical specs/tools, discovery catalog, frequently accessed utilities
- **Tier2**: Advanced analyzers, framework specs, debugging patterns
- **Tier3**: READMEs, CHANGELOG, test suites, monitoring tools

**Common Pitfalls**:

| Mistake | Impact | Fix |
|:--------|:-------|:----|
| README in tier1/tier2 | +15k tokens | Move to tier3 |
| All analyzers in tier1 | +10k tokens | Keep 2-3 critical, rest tier2 |
| User docs in bundles | Token bloat | Link to docs, don't embed |
| Python source in tier1 | ~2-3k/file | Reserve for most-used only |

**File Size Guidelines**:
- Python: ~2-3k tokens each (tier1: 2-3 files max)
- Short specs (300-600t): tier1
- Medium guides (1-3k): tier2
- Long manuals (10k+): tier3

**Strategy**: Measure first (`make qms-bundle-tokens`), tier by usage frequency

---

## Architecture Map

### Specs Directory

```
AgentQMS/specs/
├── tier1-contracts/              # Immutable contracts
│   ├── ads-specification.md
│   └── workflow-requirements.spec.md
├── tier2-framework/              # 10 consolidated specs
│   ├── ocr-engine.spec.md        # 12.6KB
│   ├── configuration.spec.md     # 9.5KB
│   ├── patterns.spec.md
│   ├── runtime.spec.md
│   ├── discovery.spec.md
│   ├── agent-infra.spec.md
│   ├── api.spec.md
│   ├── core-infra.spec.md
│   ├── constraints.spec.md
│   └── framework_specs.spec.md
└── tier3-implementation/
```

### Context Bundles

```
AgentQMS/.agentqms/plugins/context_bundles/
├── ast-debugging-tools.yaml
├── hydra-configuration.yaml
├── ocr-debugging.yaml
├── v5-domains-standard.yaml
└── ... (17 total)
```

**Discovery**: Auto-load based on task keywords
**Measurement**: `make qms-bundle-tokens`

---

## References

- [ADS v2.0 Specification](tier1-contracts/ads-specification.md)
- [Hydra V5 Configuration](tier2-framework/configuration.spec.md)
- [OCR Engine Specification](tier2-framework/ocr-engine.spec.md)
- [Discovery Tools](tier2-framework/discovery.spec.md)

---

**Target Audience**: AI Agents (Claude, Gemini, Copilot)
**Purpose**: Prevent architecture drift via constraints and rules
