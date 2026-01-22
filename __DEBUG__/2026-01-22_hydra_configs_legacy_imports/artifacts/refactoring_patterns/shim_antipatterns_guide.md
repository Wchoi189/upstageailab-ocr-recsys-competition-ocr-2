# Backward Compatibility Shims: Antipatterns and Best Practices

**Source**: Conversation analysis
**Date**: 2026-01-22
**Context**: Hydra configuration refactoring and legacy import management
**Python Manager**: uv

---

## Executive Summary

Backward compatibility shims, while marketed as safety nets during refactoring, often evolve into technical debt traps that actively sabotage codebase health. This document outlines why shims become toxic over time and provides alternatives for managing legacy code during migrations.

---

## Core Problem: Leakage of Abstraction

Shims create a **mapping between two different mental models**, leading to:

### 1. The Maintenance "Double-Tax"

When implementing a shim, you maintain:
- The new system's logic
- The old system's API
- **The translation layer between them**

**Impact**: Every new feature requires asking: *"How do I expose this through the shim?"*
**Result**: Glue code becomes more complex than actual feature logic.

### 2. The "Ghost in the Machine" (Debugging Nightmares)

Shims are notorious for swallowing or mutating stack traces.

**Obscured Failures**:
- In Hydra/OCR cases, shims may map old configuration paths to new ones
- When failures occur, error messages point to the shim's internal logic rather than the actual missing file or misconfigured class
- Encourages "passive" fixing instead of failing loudly and forcing migration

**Example**:
```python
# Shim hides the real error
try:
    new_module = import_new_path(old_path)
except ImportError:
    # Generic error - loses context
    raise ShimError("Failed to load module")
```

### 3. Logic Drift and "Zombie" Code

Over time, the new system evolves in ways the old API cannot represent.

**The Mismatch**:
- New parameter requires a boolean
- Old shim only accepts strings
- "Guesswork" logic converts strings to booleans

**The Risk**: The shim becomes a **translator that hallucinates**, making assumptions about intent that the original caller never had.

### 4. The "Incentive to Procrastinate"

**Psychological Harm**: Shims remove the "healthy friction" required to force upgrades to latest standards.

- If the shim works "well enough," migration is deprioritized forever
- Results in **Frankenstein architecture**: 40% of codebase is translation layers for code that should have been deleted years ago

---

## When Are Shims Actually "Good"?

Shims are only useful when they are **Temporary and Instrumented**:

| Feature        | Toxic Shim                  | Healthy Shim                                     |
| -------------- | --------------------------- | ------------------------------------------------ |
| **Duration**   | Permanent "legacy support." | Hard "Sunset Date" (e.g., 6 months).             |
| **Visibility** | Silent translation.         | Logs a `DeprecationWarning` every time it's hit. |
| **Complexity** | Contains its own logic.     | Only re-routes calls to the new API.             |
| **Testing**    | Ignored by unit tests.      | Has explicit "Compatibility Tests."              |

---

## Better Alternative: The "Hard Break"

### Validation Layer Pattern

Instead of a shim, implement a **Validation Layer** that:
1. Explicitly identifies "Legacy Layouts"
2. **Refuses to run** until they are updated
3. Provides clear upgrade guidance

**Benefits**:
- Moment of pain prevents lifetime of confusion
- Forces systematic migration
- Clear error messages guide developers

**Example Implementation**:
```python
# scripts/audit/migration_guard.py
import sys
from pathlib import Path

LEGACY_PATTERNS = {
    "ocr.core.metrics": "Moved to ocr.domains.{domain}.metrics",
    "ocr.detection": "Renamed to ocr.domains.detection",
}

def validate_no_legacy_imports():
    """Fail fast if legacy imports are detected."""
    for pattern, message in LEGACY_PATTERNS.items():
        if pattern_found_in_codebase(pattern):
            print(f"❌ LEGACY IMPORT DETECTED: {pattern}")
            print(f"   Migration required: {message}")
            sys.exit(1)
```

---

## The "Alias" Shim (The Only Good Shim)

If a class is used in 100+ YAML files and you don't want to update them all at once, use a **Python Alias** in the old location's `__init__.py`.

**Why This Works**:
- Lives in code, not in a hidden translation layer
- Provides deprecation warnings
- Allows gradual migration

**Example**:
```python
# ocr/core/metrics/__init__.py (Old Location)
import warnings
from ocr.domains.detection.metrics.cleval_metric import CLEvalMetric as _NewCLEvalMetric

# This allows Hydra to still find it at the old path while warning the user
CLEvalMetric = _NewCLEvalMetric
warnings.warn(
    "ocr.core.metrics.CLEvalMetric is deprecated. "
    "Use ocr.domains.detection.metrics.CLEvalMetric",
    DeprecationWarning
)
```

**Benefits**:
- Code doesn't break immediately
- Audit tools can still flag "Deprecated usage"
- Clear migration path

---

## Migration Validator Pattern

Instead of trying to make old config work with new code, write a script that:
1. Identifies legacy path configurations
2. Provides clear upgrade guide
3. Refuses to run until updated

**See Also**:
- `migration_guard_implementation.md` for complete implementation
- `validation_layer_patterns.md` for validation strategies
