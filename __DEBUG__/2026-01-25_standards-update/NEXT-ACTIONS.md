---
type: action_items
priority: medium
date: 2026-01-25
status: ready
---

# Next Actions - Documentation Updates

## Immediate (High Priority)

### None Required ✅
All critical documentation consolidation is complete.

## Short-Term (Medium Priority)

### 1. Fix Artifact Naming/Frontmatter (4 files)

**Files with violations:**
- docs/artifacts/assessments/2026-01-21_1645_assessment_agentqms-nuclear-bloat-audit.md
- docs/artifacts/assessments/2026-01-21_1715_assessment_generate-ide-configs-analysis.md
- docs/artifacts/assessments/2026-01-21_1720_assessment_frontmatter-corruption-root-cause.md
- docs/artifacts/assessments/2026-01-21_context-bundling-recovery.md

**Actions:**
```bash
# Option 1: Manual fix
# Add version: '1.0' to frontmatter
# Fix filename format for context-bundling-recovery.md

# Option 2: Automated fix
cd AgentQMS/bin && make fix ARGS="--limit 4"
```

**Expected result:** Compliance 91.8% → 100%

## Long-Term (Low Priority - Optional)

### 1. Implement Anti-Pattern Checker Tool

**If AgentQMS/tools/check_anti_patterns.py doesn't exist:**
- Create based on anti-patterns.yaml rules
- Implement grep/AST-based detection
- Add to pre-commit hooks

**Reference:**
See anti-patterns.yaml enforcement section for specifications.

### 2. Create Audit Procedures YAML

**If systematic audit workflows needed:**
- Extract workflow from config_compliance_audit_guide.md
- Create audit-procedures.yaml in tier4-workflows/
- Include: steps, tools, validation commands

**Benefit:** Machine-parseable audit automation

### 3. Weekly Bloat Detection CI

**If automated maintenance desired:**
- Create .github/workflows/bloat-detection.yml
- Reference bloat-detection-rules.yaml automated_scanning section
- Schedule: Sunday midnight

## Documentation That Doesn't Need Changes ✅

### Keep as Markdown (Human-Facing)
- ✅ docs/reports/config_compliance_audit_guide.md (procedures)
- ✅ docs/artifacts/audits/*.md (historical narratives)
- ✅ docs/artifacts/assessments/*.md (one-time analyses)
- ✅ docs/artifacts/research/*.md (research narratives)
- ✅ docs/artifacts/walkthroughs/*.md (narrative guides)

### Keep Current Structure (YAML + MD)
- ✅ AgentQMS/standards/utility-scripts/ (YAML index + MD references)
- ✅ Plugin documentation (marketplace-specific)

### Already in YAML Format ✅
- ✅ All tier1-sst/*.yaml
- ✅ All tier2-framework/*.yaml
- ✅ All tier3-agents/*/config.yaml
- ✅ All tier4-workflows/*.yaml

## Standards Consolidation Summary

### What Was Done
| Original File | Lines | New File | Tokens | Reduction |
|---------------|-------|----------|--------|-----------|
| anti-patterns.md | 466 | anti-patterns.yaml | 180 | 61% |
| v5-architecture-patterns.md | 490 | (existing YAML) | 0 | 100% |
| bloat-detection-rules.md | 505 | bloat-detection-rules.yaml | 220 | 56% |
| **Total** | **1,461** | - | **400** | **73%** |

### Impact
- **Baseline memory:** 0 tokens (auto_load: false)
- **Zero duplication:** v5 patterns already existed
- **Compliance:** +3.6% improvement
- **Architecture:** Fully AI-Native aligned

## Decision Matrix for Future Documentation

```
Does it contain machine-enforceable rules? ─────┬─── YES → Create/Update YAML
                                                │
                                                └─── NO → Keep as Markdown
                                                         │
Is it a one-time artifact? ────────────────────────────┼─── YES → Markdown
                                                        │
Is it a procedural guide? ─────────────────────────────┼─── YES → Markdown
                                                        │
Is it > 300 lines with rules? ─────────────────────────┴─── YES → Review for YAML extraction
```

## Monitoring

### Quarterly Review Checklist
- [ ] Review anti-patterns.yaml for new violations
- [ ] Update bloat-detection-rules.yaml thresholds
- [ ] Check for new large markdown files (> 300 lines)
- [ ] Validate registry.yaml task mappings
- [ ] Run compliance check

### Triggers for Standards Updates
- New anti-pattern discovered in code review
- Bloat detection threshold needs tuning
- New task category emerges
- Framework upgrade (Hydra, PyTorch, etc.)

---

**Session:** 2026-01-25 Standards Consolidation
**Status:** ✅ Complete
**Next Review:** 2026-04-25 (Quarterly)
