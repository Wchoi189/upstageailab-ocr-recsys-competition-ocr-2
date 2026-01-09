---
title: Context Bundling Strategy - Low-Memory Design & Domain Specialization
date: 2026-01-09 20:30 (KST)
type: design
category: architecture
status: active
version: '1.0'
ads_version: '1.0'
related_artifacts:
  - 2026-01-09_2020_implementation_plan_context-system-consolidation-completed.md
generated_artifacts: []
tags:
  - agentqms
  - context
  - bundles
  - memory-optimization
  - domain-specialization
  - glob-patterns
---

# Context Bundling Strategy: Low-Memory Design & Domain Specialization

## Executive Summary

**Key Findings**:
1. ✅ **Context bundles ≠ glob patterns** (complementary, not competing)
2. ✅ **Tiered approach alone is insufficient** for memory-constrained use
3. ✅ **Domain-specialized bundles are essential** for production LLM usage
4. ✅ **Your proposed 5 bundles are highly valuable** for focused contexts
5. ✅ **OCR structure supports clean domain separation** (feature-first + shared core)

**Recommendation**: Create **two-level bundle hierarchy**:
- **Generic bundles** (pipeline-development, ocr-experiment) for learning/exploration
- **Domain-specific bundles** (detection, recognition, layout, KIE, Hydra) for production

---

## Part 1: Glob Patterns vs Context Bundles

### Understanding the Distinction

**Glob Patterns** (file-level):
```yaml
# Within a bundle definition
files:
  - path: ocr/features/**/*.py           # Glob pattern
  - path: configs/domain/*.yaml          # Glob pattern
  - path: ocr/core/inference/*.py        # Glob pattern
```

**Context Bundles** (collection-level):
```yaml
# Bundle = curated group of contexts with metadata
name: ocr-text-detection
description: All files needed for text detection tasks
tiers:
  tier1: [critical files]
  tier2: [detailed files]
```

### Relationship

```
┌─────────────────────────────────────────────────┐
│ Context Bundle (High Level)                     │
│ Example: ocr-text-detection                    │
│                                                 │
│  ├─ Tier 1 (Essential)                         │
│  │  ├─ ocr/features/detection/**/*.py (GLOB)   │
│  │  └─ configs/domain/detection.yaml           │
│  │                                             │
│  └─ Tier 2 (Detailed)                          │
│     ├─ ocr/core/models/*.py (GLOB)             │
│     └─ tests/test_detection.py                 │
│                                                 │
└─────────────────────────────────────────────────┘
```

**Answer**: Context bundles **use** glob patterns, but bundles are the **organization container**.

---

## Part 2: Memory Footprint Analysis

### The Problem With Generic Bundles

**Current `pipeline-development.yaml`**:
- Tier1: 8 files
- Tier2: 12 files
- Tier3: 10 files
- **Total: 30 file