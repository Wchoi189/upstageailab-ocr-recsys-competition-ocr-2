# Migration Guard Implementation

**Source**: Conversation analysis
**Date**: 2026-01-22
**Context**: Systematic migration validation for Hydra refactoring
**Python Manager**: uv

---

## Overview

The Migration Guard is a **pre-execution validation script** that compares the "State of the Config" against the "State of the Disk" to prevent runtime errors from legacy imports and outdated Hydra targets.

---

## Core Implementation

### Basic Migration Guard

```python
# scripts/audit/migration_guard.py
import os
import subprocess
import yaml
from pathlib import Path

# Ground Truth: Where things SHOULD be after refactor
EXPECTED_MAPPINGS = {
    "ocr.core.metrics": "ocr/core/evaluation/metrics.py",
    "ocr.domains.detection.utils.logging": "ocr/core/utils/logging.py",
    "ocr.core.lightning": "ocr/core/lightning/module_factory.py"
}

def check_editable_install():
    """Checks if the 'ocr' package is installed in editable mode."""
    import ocr
    origin = ocr.__file__
    if "site-packages" in origin:
        print(f"❌ CRITICAL: Ghost Code detected! Using {origin}")
        print("👉 Fix: uv pip install -e .")
        return False
    else:
        print(f"✅ Environment: Local workspace active ({origin})")
        return True

def validate_hydra_targets():
    """Uses yq to find targets and checks if they exist in Python."""
    print("🔍 Auditing Hydra Targets...")
    cmd = 'find configs/ -name "*.yaml" -exec yq \'.. | select(has("_target_")) | ._target_\' {} +'

    try:
        targets = subprocess.check_output(cmd, shell=True).decode().splitlines()
    except subprocess.CalledProcessError:
        print("⚠️  Warning: Could not extract Hydra targets")
        return True

    broken_targets = []
    for target in set(targets):
        if not target or target == "null":
            continue
        # Check if target is importable
        module_path = target.rsplit(".", 1)[0]
        try:
            __import__(module_path)
        except (ImportError, ModuleNotFoundError):
            broken_targets.append(target)

    if broken_targets:
        print(f"❌ Found {len(broken_targets)} broken Hydra targets:")
        for target in broken_targets[:10]:  # Show first 10
            print(f"   - {target}")
        return False
    else:
        print("✅ All Hydra targets are valid")
        return True

def check_recursive_instantiation():
    """Check for missing _recursive_=False in Hydra instantiate calls."""
    print("🔍 Checking for recursive instantiation traps...")
    cmd = 'grep -r "hydra.utils.instantiate" ocr/ | grep -v "_recursive_=False" || true'

    try:
        result = subprocess.check_output(cmd, shell=True).decode()
        if result.strip():
            print("⚠️  WARNING: Potential recursive instantiation traps found:")
            for line in result.strip().split('\n')[:5]:
                print(f"   {line}")
            return False
        else:
            print("✅ Hydra recursion safety looks good")
            return True
    except subprocess.CalledProcessError:
        return True

if __name__ == "__main__":
    print("🛡️  Starting Migration Validation...")
    print("=" * 60)

    checks = [
        check_editable_install(),
        validate_hydra_targets(),
        check_recursive_instantiation()
    ]

    print("=" * 60)
    if all(checks):
        print("✅ All migration checks passed!")
        exit(0)
    else:
        print("❌ Migration validation failed. Fix issues before proceeding.")
        exit(1)
```

---

## Runtime Assert Pattern

Add a **Hard Break** in your training entry point to prevent running with stale code:

```python
# runners/train.py
import ocr
import sys

# Runtime assertion to prevent "Ghost Code" execution
assert "site-packages" not in ocr.__file__, (
    "🛑 STOP: You are editing code but running an installed package copy!\n"
    f"   Current location: {ocr.__file__}\n"
    "   Fix: uv pip install -e ."
)
```

**Benefits**:
- Creates a "Hard Break" that prevents passive reactions to stale code
- Fails immediately with clear error message
- Forces developer to fix environment before proceeding

---

## Pre-Flight Check Script

Comprehensive validation script that runs before any training or testing:

```bash
#!/bin/bash
# scripts/preflight.sh
set -e

echo "🔍 [1/4] Checking environment status..."
PYTHON_ORIGIN=$(uv run python -c "import ocr; print(ocr.__file__)")

if [[ "$PYTHON_ORIGIN" == *"site-packages"* ]]; then
    echo "❌ ERROR: Ghost Code Detected. Running from $PYTHON_ORIGIN"
    echo "Fixing: Re-installing in editable mode..."
    uv pip install -e .
else
    echo "✅ Environment: Editable install verified."
fi

echo "🔍 [2/4] Running Master Audit..."
uv run python scripts/audit/master_audit.py > audit_results.txt

echo "🔍 [3/4] Validating Hydra Recursion Safety..."
# Check if factories use _recursive_=False
BAD_INSTANCES=$(grep -r "hydra.utils.instantiate" ocr/ | grep -v "_recursive_=False" || true)
if [ ! -z "$BAD_INSTANCES" ]; then
    echo "⚠️  WARNING: Potential Recursive Instantiation traps found:"
    echo "$BAD_INSTANCES"
else
    echo "✅ Hydra: Recursion safety looks good."
fi

echo "📊 [4/4] Summary:"
grep "🚨" audit_results.txt || echo "No critical anomalies found."
```

---

## Integration with CI/CD

### GitHub Actions Example

```yaml
# .github/workflows/migration-check.yml
name: Migration Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python with uv
        uses: astral-sh/setup-uv@v1

      - name: Install dependencies
        run: uv pip install -e .

      - name: Run migration guard
        run: uv run python scripts/audit/migration_guard.py

      - name: Run pre-flight checks
        run: bash scripts/preflight.sh
```

---

## Usage Patterns

### Manual Validation

```bash
# Before starting work
uv run python scripts/audit/migration_guard.py

# If checks fail, fix issues:
uv pip install -e .  # Fix editable install
# Then re-run validation
```

### Automated Validation

```bash
# Add to Makefile
.PHONY: validate
validate:
	@bash scripts/preflight.sh

# Use before training
make validate && uv run python runners/train.py
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
uv run python scripts/audit/migration_guard.py || {
    echo "Migration validation failed. Commit blocked."
    exit 1
}
```

---

## Best Practices

1. **Run Before Every Training Session**: Ensures environment is clean
2. **Integrate with CI/CD**: Catch issues before they reach production
3. **Update EXPECTED_MAPPINGS**: Keep ground truth current with refactoring
4. **Log Results**: Save audit outputs for debugging
5. **Fail Fast**: Don't allow execution with known issues

---

## See Also

- `shim_antipatterns_guide.md` - Why validation is better than shims
- `hydra_target_validation.md` - Detailed Hydra-specific validation strategies
- `ai_instruction_patterns.md` - How to present validation results to AI agents
