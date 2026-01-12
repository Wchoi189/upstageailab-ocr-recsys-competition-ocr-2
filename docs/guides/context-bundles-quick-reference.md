# Context Bundles Quick Reference

## 📋 Bundle Selection Guide

### For Text Detection Tasks
```
Bundle: ocr-text-detection
Use: "implement text detection algorithm"
Memory: ~120 KB | 3% budget
Keywords: detection, text, bbox, localization
```

### For Text Recognition Tasks
```
Bundle: ocr-text-recognition
Use: "fix character recognition issues"
Memory: ~150 KB | 3.5% budget
Keywords: recognition, character, text, ocr, charset
```

### For Layout Analysis
```
Bundle: ocr-layout-analysis
Use: "analyze page layout and reading order"
Memory: ~140 KB | 3.3% budget
Keywords: layout, structure, reading-order, page-segmentation
```

### For Key-Value Extraction
```
Bundle: ocr-information-extraction
Use: "extract key-value information from documents"
Memory: ~130 KB | 3% budget
Keywords: kie, extraction, key-value, entity, information
```

### For Configuration & Experiments
```
Bundle: hydra-configuration
Use: "configure hydra experiments"
Memory: ~100 KB | 2.4% budget
Keywords: hydra, configuration, config, experiment, yaml
```

---

## 🔍 How Context Suggestion Works

```bash
# 1. User describes task
python suggest_context.py "implement text detection"

# 2. System matches keywords
OCR-TEXT-DETECTION contains: "detection", "text"
Task contains: "implement", "text", "detection"
Match score: 2 keywords × 2 points = 4 points

# 3. System ranks bundles by score
1. ocr-text-detection (score: 4) ✅ HIGHEST
2. pipeline-development (score: 1)
3. ocr-experiment (score: 0)

# 4. Top bundle is recommended
```

---

## 📊 Bundle Memory Footprint

| Bundle                     | Tier1 Size | Est. KB | % Context |
| -------------------------- | ---------- | ------- | --------- |
| ocr-text-detection         | ~5 files   | 120 KB  | 3.0%      |
| ocr-text-recognition       | ~5 files   | 150 KB  | 3.5%      |
| ocr-layout-analysis        | ~5 files   | 140 KB  | 3.3%      |
| ocr-information-extraction | ~5 files   | 130 KB  | 3.0%      |
| hydra-configuration        | ~4 files   | 100 KB  | 2.4%      |
| pipeline-development       | ~8 files   | 400 KB  | 10.0%     |

**Savings**: Domain bundles use 60-70% less context than generic

---

## 🎯 When to Use Each Bundle Type

### Generic Bundles (Learning)
- New team member learning architecture
- Overview of full system
- Understanding relationships

### Domain-Specific Bundles (Production)
- Implementing specific OCR feature
- Debugging domain-specific issues
- Optimizing performance in one area
- **Recommended for AI agents** (memory efficient)

---

## 🚀 Quick Usage

```bash
# Suggest bundle for task
uv run python suggest_context.py "your task description"

# Load specific bundle programmatically
from AgentQMS.tools.core.context_bundle import get_context_bundle
files = get_context_bundle("", bundle_name="ocr-text-detection")

# List all bundles
uv run python suggest_context.py --list-bundles
```

---

## 💡 Pro Tips

1. **Start with Tier1**: Contains critical code only (~3% context budget)
2. **Add Tier2 if needed**: For implementation details and utilities
3. **Reference Tier3 rarely**: Tests and examples are optional
4. **Use domain bundles for agents**: 67% memory reduction
5. **Use generic for humans**: Better for learning and overview

---

## 🔗 Related Files

- Design Document: `docs/artifacts/design_documents/2026-01-09_2030_design-context-bundling-strategy-low-memory.md`
- Implementation Plan: `docs/artifacts/implementation_plans/2026-01-09_2020_implementation_plan_context-system-consolidation-completed.md`
- Bundle Definitions: `AgentQMS/.agentqms/plugins/context_bundles/`

---

## ❓ FAQ

**Q: Should I load tier1 or all tiers?**
A: Start with tier1. Load tier2 if you need implementation details.

**Q: What's the difference between bundles?**
A: Generic = full system overview. Domain = specific feature only.

**Q: How many files will be loaded?**
A: Tier1: 4-8 files (~100-150 KB). Tier2: 6-12 files (add 100-200 KB).

**Q: Can I combine bundles?**
A: Not yet, but future enhancement planned.

**Q: Which bundle for new developers?**
A: `pipeline-development` (tier1) for architecture overview.

**Q: Which bundle for implementing detection?**
A: `ocr-text-detection` (tier1 + tier2 if needed).

---

**Status**: ✅ **11 context bundles active and validated**
**Suggestion System**: ✅ **Working with keyword matching**
**Memory Optimization**: ✅ **60-70% reduction achieved**
