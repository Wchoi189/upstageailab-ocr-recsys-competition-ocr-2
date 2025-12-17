---
type: architecture
component: api
status: current
version: "1.0"
last_updated: "2025-12-15"
---

# API Decoupling from Streamlit

**Purpose**: FastAPI playground API decoupled from Streamlit; enables Next.js frontend without Streamlit dependencies or initialization overhead.

---

## Architecture

| Component | Endpoints | Streamlit Dependency |
|-----------|-----------|---------------------|
| **Playground API** | `/api/commands/schemas`, `/api/commands/schemas/{schema_id}`, `/api/commands/build`, `/api/commands/recommendations` | ❌ None |
| **Next.js Console** | Calls FastAPI via Next.js API routes | ❌ None |
| **Legacy Streamlit** | Can use same API endpoints or internal functions | ✅ Optional |

---

## Decoupling Strategy

| Approach | Implementation | Benefit |
|----------|----------------|---------|
| **Pure Functions Module** | Created `ui/utils/override_compute.py` (Streamlit-free); API imports from `override_compute` | API doesn't trigger Streamlit init |
| **Lazy Package Imports** | `ui/apps/command_builder/__init__.py`, `ui/apps/command_builder/services/__init__.py` use lazy imports | Prevents circular dependencies |
| **Dependency Clarification** | `ConfigParser` doesn't import Streamlit; only triggers registry/model init | Framework-independent |

---

## Import Chain Evolution

**Before** (Streamlit import):
```python
from ui.utils.ui_generator import compute_overrides
# → Imports ui_generator → imports streamlit → initializes Streamlit
```

**After** (Streamlit-free):
```python
from ui.utils.override_compute import compute_overrides
# → Pure function, no Streamlit dependency
```

---

## Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Startup Time** | 10-15s (Streamlit + registry) | <2s (registry only on first call) | 5-7x faster |
| **Memory** | Streamlit session state at startup | No Streamlit overhead | Reduced |
| **Cold Start** | Immediate API start + lazy load on first call | Cached with `@lru_cache` | Instant subsequent calls |

---

## Dependencies

| Component | Imports | Internal Dependencies |
|-----------|---------|----------------------|
| **Playground API** | FastAPI, override_compute | ConfigParser (Streamlit-free) |
| **override_compute** | None (pure functions) | None |
| **ConfigParser** | Registry, model init | No Streamlit |

---

## Constraints

- **Lazy Loading**: First API call triggers one-time registry initialization
- **LRU Cache**: `@lru_cache` decorator required for performance
- **Import Isolation**: API must import from `override_compute`, not `ui_generator`

---

## Backward Compatibility

**Status**: Maintained for legacy Streamlit apps

**Breaking Changes**: None (legacy apps can still use internal functions)

**Compatibility Matrix**:

| Client | API Access | Streamlit Dependency | Status |
|--------|------------|---------------------|--------|
| Next.js Console | ✅ FastAPI | ❌ None | ✅ Full support |
| Legacy Streamlit | ✅ FastAPI or internal | ✅ Optional | ✅ Full support |

---

## References

- [System Architecture](system-architecture.md)
- [Backend Pipeline Contract](../backend/api/backend-pipeline-contract.md)
```bash
grep -r "import streamlit\|from streamlit" services/playground_api
# Should return no results
```

### Test API Import
```python
# Should not trigger Streamlit initialization
from services.playground_api.routers.command_builder import router
# No Streamlit warnings should appear
```

### Test Override Compute
```python
# Should work without Streamlit
from ui.utils.override_compute import compute_overrides
# Works in both FastAPI and Streamlit contexts
```

## Related Files

- `ui/utils/override_compute.py` - Streamlit-free override computation
- `ui/utils/config_parser.py` - Pure config parser (no Streamlit)
- `services/playground_api/routers/command_builder.py` - API router (Streamlit-free imports)
- `ui/apps/command_builder/__init__.py` - Lazy imports to prevent deadlocks
- `docs/performance/fastapi-startup-optimization.md` - Performance optimization details

## Future Improvements

1. **Extract More Pure Functions** - Move other UI-agnostic logic to separate modules
2. **Metadata Caching** - Cache registry metadata in JSON to avoid initialization
3. **API-First Design** - Design new features as API-first, then add UI clients
