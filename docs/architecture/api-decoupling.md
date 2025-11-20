# API Decoupling from Streamlit

## Overview

The FastAPI playground API (`services/playground_api`) is fully decoupled from Streamlit. This allows the Next.js frontend and other clients to use the API without any Streamlit dependencies or initialization overhead.

## Architecture

### Universal API Endpoints

All API endpoints in `services/playground_api` are Streamlit-free:

- `/api/commands/schemas` - List available command schemas
- `/api/commands/schemas/{schema_id}` - Get schema details
- `/api/commands/build` - Build CLI command from form values
- `/api/commands/recommendations` - Get architecture recommendations

### Frontend Clients

**Next.js Playground Console** (`apps/playground-console`):
- Calls FastAPI endpoints via Next.js API routes
- No Streamlit dependency
- Zero Streamlit initialization overhead

**Legacy Streamlit Apps** (`ui/apps/`):
- Can use the same API endpoints
- Or use internal functions directly (for Streamlit-specific features)

## Implementation Details

### Decoupling Strategy

**1. Pure Functions Module**
- Created `ui/utils/override_compute.py` (Streamlit-free)
- Extracted `compute_overrides()` function used by API
- API imports from `override_compute`, not `ui_generator`

**2. Lazy Package Imports**
- `ui/apps/command_builder/__init__.py` - lazy imports to avoid Streamlit app loading
- `ui/apps/command_builder/services/__init__.py` - lazy imports to prevent circular dependencies

**3. Dependency Clarification**
- `ConfigParser` does NOT import Streamlit
- Only triggers registry/model initialization
- Can be used independently of UI framework

### Import Chain Example

**Before (Caused Streamlit Import):**
```python
# In API router
from ui.utils.ui_generator import compute_overrides
# → Imports ui_generator → imports streamlit → initializes Streamlit
```

**After (Streamlit-Free):**
```python
# In API router
from ui.utils.override_compute import compute_overrides
# → Pure function, no Streamlit dependency
```

## Performance Benefits

### Startup Time
- **Before:** 10-15 seconds (Streamlit + registry initialization)
- **After:** < 2 seconds (only registry initialization on first API call)

### Memory Usage
- **Before:** Streamlit session state loaded at startup
- **After:** No Streamlit overhead until Streamlit app is actually used

### Cold Start
- API starts immediately
- First API call triggers lazy loading (one-time cost)
- Subsequent calls are instant (cached with `@lru_cache`)

## Verification

### Check for Streamlit Imports
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

