# AI Workflow Efficiency Survey

**Date:** 2026-01-05
**Type:** Assessment
**Tags:** optimization, documentation, ai-agents

---

## Documentation Effectiveness

### 1. Current Docs Usefulness for AI Agents

| Rating | Area                 | Notes                                 |
| ------ | -------------------- | ------------------------------------- |
| ⭐⭐⭐⭐   | `AGENTS.yaml`        | Excellent entrypoint, clear structure |
| ⭐⭐⭐    | AgentQMS standards   | Good but verbose, needs condensing    |
| ⭐⭐     | Code docstrings      | Inconsistent, often missing           |
| ⭐      | Config documentation | Hydra patterns undocumented           |

**Example pain point:** `train_v2.yaml` defaults are hard to trace without the AST debug toolkit.

### 2. Runtime Information Discovery Difficulty

**High friction areas:**
- Finding which config keys override what (Hydra composition order)
- Locating component registration (scattered `register_*` calls)
- Dataset path resolution (`${paths.data}` interpolation chain)

### 3. Consistently Missing Information

| Missing                     | Impact                               |
| --------------------------- | ------------------------------------ |
| **Component contracts**     | `__init__` signatures not documented |
| **Data format specs**       | LMDB schema only in code, not docs   |
| **Config precedence rules** | Caused BUG_003                       |

---

## Project Organization

### 4. Tiered Schema Assessment

```
✅ Tier 1 (SST): Clear, stable
✅ Tier 2 (Framework): Good coverage
⚠️ Tier 3 (Agents): Outdated agent configs
❌ Tier 4 (Workflows): Sparse, incomplete
```

### 5. Organizational Patterns Causing Confusion

| Pattern                          | Issue                          |
| -------------------------------- | ------------------------------ |
| `ocr/data/` vs `ocr/datasets/`   | Overlapping concerns           |
| `configs/data/` nested structure | Deep paths slow search         |
| `scripts/` mega-directory        | 24 files, 22 subdirs, no index |

### 6. Structural Elements Slowing Development

1. **No `__init__.py` exports** → Can't quickly see module API
2. **Archive pollution** → `archive/` appears in search results
3. **Scattered configs** → `model/architectures/` vs `model/` inconsistent

---

## Improvement Priorities

### 7. Top 3 Workflow Pain Points

1. **Config debugging** → Hydra merge order opaque without AST toolkit
2. **Entry point discovery** → Training runners don't surface available configs
3. **Dataset registration** → Manual registry vs Hydra `_target_` inconsistent

### 8. Highest Impact Improvements

| Improvement                    | Effort | Impact |
| ------------------------------ | ------ | ------ |
| Centralize data format schemas | Low    | ⭐⭐⭐⭐⭐  |
| Add `__all__` to key modules   | Low    | ⭐⭐⭐⭐   |
| Flatten `configs/` structure   | Medium | ⭐⭐⭐⭐   |
| Index `scripts/` directory     | Low    | ⭐⭐⭐    |

### 9. Next Focus Area

> [!IMPORTANT]
> **Recommended:** Create `ocr/data/schemas/` with Pydantic models for all data contracts (LMDB, detection, KIE). This eliminates the "data format not documented" gap.

---

## Actionable Recommendations

### Immediate (This Session)

1. Add `ocr/datasets/__init__.py` with exports
2. Document LMDB schema in `dataset-catalog.yaml` ✅ (done)
3. Add `scripts/README.md` index

### Short-term (Next Sprint)

1. Flatten `configs/data/` to 2 levels max
2. Create component contract docstrings (follow coding-standards.yaml)
3. Deprecate `ocr/data/datasets/` in favor of `ocr/datasets/`

### Medium-term

1. Auto-generate config documentation from YAML
2. Add `--list-configs` to training runner
3. Implement dataset discovery via Hydra structured configs
